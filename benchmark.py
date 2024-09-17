import re
import subprocess
import time
from collections import defaultdict
from enum import Enum
from os import path
from threading import Timer
from typing import Optional

import numpy

import utils
from model import (
    BenchmarkReport,
    ErrorReport,
    GarbageCollectorReport,
)


class BENCHMARK_GROUP(Enum):
    RENAISSANCE = "Renaissance"
    DACAPO = "DaCapo"


_dir = path.dirname(__file__)
_BENCHMARK_PATH = f"{_dir}/benchmark_apps"
_benchmark_paths = {
    BENCHMARK_GROUP.RENAISSANCE.value: f"{_BENCHMARK_PATH}/renaissance-gpl-0.15.0.jar",
    BENCHMARK_GROUP.DACAPO.value: f"{_BENCHMARK_PATH}/dacapo-23.11-chopin.jar",
}


def _get_benchmarks(benchmark_group: BENCHMARK_GROUP) -> list[str]:
    match benchmark_group:
        case BENCHMARK_GROUP.RENAISSANCE:
            result = subprocess.run(
                [
                    "java",
                    "-jar",
                    _benchmark_paths[benchmark_group.value],
                    "--raw-list",
                ],
                capture_output=True,
                text=True,
            )
            assert result.stderr == "", result.stderr
            return result.stdout.splitlines()

        case BENCHMARK_GROUP.DACAPO:
            result = subprocess.run(
                [
                    "java",
                    "-jar",
                    _benchmark_paths[benchmark_group.value],
                    "--list-benchmarks",
                ],
                capture_output=True,
                text=True,
            )
            assert result.stderr == "", result.stderr
            return result.stdout.split()
        case _:
            raise AssertionError("Benchmark group is not supported")


def _get_benchmark_command(
    benchmark_group: BENCHMARK_GROUP,
    benchmark: str,
    gc: str,
    heap_size: str,
    iterations: int,
) -> list[str]:
    bench_path = _benchmark_paths[benchmark_group.value]
    command = [
        "java",
        f"-XX:+Use{gc}GC",
        f"-Xms{heap_size}m",
        f"-Xmx{heap_size}m",
        f"-Xlog:gc*,safepoint:file={utils.get_benchmark_log_path(gc, benchmark_group.value, benchmark, heap_size)}::filecount=0",
        "-jar",
        bench_path,
        benchmark,
    ]

    match benchmark_group:
        case BENCHMARK_GROUP.RENAISSANCE:
            command.extend(["-r", f"{iterations}", "--no-forced-gc"])
        case BENCHMARK_GROUP.DACAPO:
            command.extend(["-n", f"{iterations}", "--no-pre-iteration-gc"])
            # TODO: change to capture output only once https://github.com/dacapobench/dacapobench/issues/265
            # is fixed
            p = subprocess.run(
                ["java", "-jar", bench_path, benchmark, "--sizes"],
                # capture_output=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if "large" in p.stdout:
                command.extend(["-s", "large"])
        case _:
            raise AssertionError(f"{benchmark_group} not supported")
    return command


def run_benchmark(
    benchmark_group: BENCHMARK_GROUP,
    benchmark: str,
    gc: str,
    heap_size: str,
    iterations: int,
    jdk: str,
    timeout: int,
) -> BenchmarkReport:
    """
    Args:
        timeout: int -> value in seconds to interrupt benchmark
    Returns:
        BenchmarkReport
    """

    def kill_process(process: subprocess.Popen[bytes], cmd: str):
        print(f"Killing command: {cmd} due to timeout")
        process.kill()

    benchmark_command = _get_benchmark_command(
        benchmark_group, benchmark, gc, heap_size, iterations
    )
    print(
        f"[{benchmark_group.value}]: Running benchmark {benchmark} with:\n\tGC: {gc}\n\tHeap Size: {heap_size}\n\tIterations: {iterations=}"
    )
    process = subprocess.Popen(
        benchmark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    timer = Timer(timeout, kill_process, (process, benchmark_command))
    time_start = time.time_ns()
    pid = process.pid
    cpu_usage_stats = []
    cpu_time_stats = []
    io_time_stats = []

    # TODO: see better polling method than sleeping 1 seconds for throughput
    timer.start()
    while process.poll() is None:
        # subprocess.run with capture_output doesn't seem to capture the whole output when
        # using top with -1 flag
        p = subprocess.run(
            ["top", "-bn", "1", "-p", f"{pid}"], capture_output=True, text=True
        )

        lines = p.stdout.splitlines()
        # NOTE: from man top(1)
        # us, user    : time running un-niced user processes
        # wa, IO-wait : time waiting for I/O completion
        # %Cpu(s): 15.1 us,  2.2 sy,  0.0 ni, 81.2 id,  0.0 wa,  0.0 hi,  1.6 si,  0.0 st
        us, wa = map(float, re.findall("(\\d+.\\d+) us.*(\\d+.\\d+) wa", lines[2])[0])
        io_time_stats.append(wa)
        cpu_time_stats.append(us)

        lines = lines[-2:]
        assert lines[0].split()[8] == "%CPU"
        cpu_usage = round(float(lines[1].split()[8]) / utils.get_cpu_count(), 1)
        cpu_usage_stats.append(cpu_usage)
        time.sleep(0.1)
    timer.cancel()
    throughput = time.time_ns() - time_start

    cpu_usage_avg = round(float(numpy.mean(cpu_usage_stats)), 1)
    cpu_time_avg = round(float(numpy.mean(cpu_time_stats)), 1)
    io_time_avg = round(float(numpy.mean(io_time_stats)), 1)
    p90_io = float(round(numpy.percentile(io_time_stats, 90), 2))

    print(
        f"{cpu_usage_avg=} {cpu_time_avg=} {io_time_avg=} {p90_io=} and {throughput=}"
    )

    if process.returncode != 0:
        error = process.stderr.read().decode() if process.stderr is not None else ""
        print(f"Error: {error}")
        result = BenchmarkReport.build_benchmark_error(
            gc,
            benchmark_group.value,
            benchmark,
            heap_size,
            cpu_usage_avg,
            cpu_time_avg,
            io_time_avg,
            p90_io,
            jdk,
            process.returncode,
            error,
        )
    else:
        print("Success")
        result = BenchmarkReport.build_benchmark_result(
            gc,
            benchmark_group.value,
            benchmark,
            heap_size,
            cpu_usage_avg,
            cpu_time_avg,
            io_time_avg,
            p90_io,
            throughput,
            jdk,
        )
    result.save_to_json()
    return result


def run_benchmark_groups(
    gc: str,
    heap_size: str,
    iterations: int,
    jdk: str,
    timeout: int,
    benchmark_groups: Optional[list[BENCHMARK_GROUP]],
) -> list[BenchmarkReport]:
    """Run all benchmark groups or only the ones
    specified in `benchmark_groups`

    Args:
        `gc`: Garbage collector used when running the benchmarks
        `heap_size`: Heap size that the JVM will use
        `iterations`: Number of iterations to run the benchmark
        `jdk`: Java version
        `timeout`: Time in seconds to stop a benchmark from running
        `benchmark_groups`: list[BENCHMARK_GROUP] | None -> If you want to run a
        subset of the benchmarks

    Returns:
        list[BenchmarkReport]
    """
    benchmark_results: list[BenchmarkReport] = []
    groups: list[BENCHMARK_GROUP] = benchmark_groups or [*BENCHMARK_GROUP]

    for benchmark_group in groups:
        benchmarks = _get_benchmarks(benchmark_group)

        for benchmark in benchmarks:
            result = run_benchmark(
                benchmark_group, benchmark, gc, heap_size, iterations, jdk, timeout
            )

            benchmark_results.append(result)
    return benchmark_results


def run_benchmarks(
    iterations: int,
    jdk: str,
    garbage_collectors: list[str],
    skip_benchmarks: bool,
    benchmark_groups: list[BENCHMARK_GROUP],
) -> dict[str, dict[str, list[BenchmarkReport]]]:
    # { gc: heap_size: { list[BenchmarkReport } }
    benchmark_results: dict[str, dict[str, list[BenchmarkReport]]] = defaultdict(
        lambda: defaultdict(list)
    )
    heap_sizes: list[str] = utils.get_heap_sizes()

    failed_benchmarks: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for gc in garbage_collectors:
        # benchmark_results[gc] = {}
        for heap_size in heap_sizes:
            # if heap_size not in benchmark_results[gc]:
            # benchmark_results[gc][heap_size] = []

            if skip_benchmarks:
                benchmark_results[gc][heap_size] = utils.load_benchmark_results(
                    gc, heap_size, jdk
                )
            else:
                benchmark_results[gc][heap_size].extend(
                    run_benchmark_groups(
                        gc,
                        heap_size,
                        iterations,
                        jdk,
                        iterations * 60,
                        benchmark_groups,
                    )
                )

            for result in benchmark_results[gc][heap_size]:
                if not result.is_successfull():
                    # if result.heap_size not in failed_benchmarks:
                    #     failed_benchmarks[result.heap_size] = {}
                    # if result.benchmark_name not in failed_benchmarks[result.heap_size]:
                    #     failed_benchmarks[result.heap_size][result.benchmark_name] = []

                    failed_benchmarks[result.heap_size][result.benchmark_name].append(
                        (
                            result.garbage_collector,
                            result.error,  # type: ignore -> error is not None in case of a failed benchmark
                        )
                    )

    # using list to avoid modifying dict while iterating
    for gc in list(benchmark_results):
        heap_size = None
        for heap_size, results in list(benchmark_results[gc].items()):
            valid_results = [
                el
                for el in results
                if failed_benchmarks.get(el.heap_size, {}).get(el.benchmark_name)
                is None
            ]

            if len(valid_results) == 0:
                del benchmark_results[gc][heap_size]
                continue

            benchmark_results[gc][heap_size] = valid_results
            assert all(
                el.is_successfull() for el in valid_results
            ), "All benchmarks should be successfull"

        if len(benchmark_results[gc]) > 0:
            GarbageCollectorReport.build_garbage_collector_result(
                benchmark_results[gc]
            ).save_to_json()
        else:
            f"Garbage Collector {gc} doesn't have successfull benchmarks."
            del benchmark_results[gc]

    if len(failed_benchmarks) > 0:
        error_report = ErrorReport(jdk, failed_benchmarks)
        error_report.save_to_json()

    return benchmark_results
