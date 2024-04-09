from os import path
import re
import subprocess
import time
from enum import Enum
from threading import Timer
from typing import Optional

import numpy

import utils
from model import (
    BenchmarkResult,
    GarbageCollectorResult,
    _ErrorReport,
)


Benchmark_Stats = tuple[bool, float, float, int, Optional[tuple[int, str]]]
"""(cpu_intensive, average_cpu_usage_percentage, average_io_percentage, throughput, error)
cpu_intensive: bool -> True if application is cpu_bound, False otherwise
average_cpu_usage_percentage: float -> average cpu usage percentage of the benchmark
average_io_percentage: float -> average IO percentage of the benchmark
throughput: int -> Time in nanoseconds. Equivalent to program execution time
error: Optional[tuple[int, str]] -> Application return code and error message when application fails"""


class BENCHMARK_GROUP(Enum):
    RENAISSANCE = "Renaissance"
    DACAPO = "DaCapo"


_dir = path.dirname(__file__)
_BENCHMARK_PATH = f"{_dir}/benchmark_apps"
_benchmark_paths = {
    BENCHMARK_GROUP.RENAISSANCE.value: f"{_BENCHMARK_PATH}/renaissance-gpl-0.15.0.jar",
    BENCHMARK_GROUP.DACAPO.value: f"{_BENCHMARK_PATH}/dacapo/dacapo-23.11-chopin.jar",
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


def run_benchmark(benchmark_command: list[str], timeout: int) -> Benchmark_Stats:
    """
    Args:
        benchmark_command: str -> command to run benchmark
        timeout: int -> value in seconds to interrupt benchmark
    Returns:
        BenchmarkStats
    """

    process = subprocess.Popen(
        benchmark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    timer = Timer(timeout, process.kill)
    time_start = time.time_ns()
    pid = process.pid
    cpu_stats = []
    io_stats = []

    # TODO: see better polling method than sleeping 1 seconds for throughput
    timer.start()
    is_cpu_intensive = 0  # NOTE: True if positive, false otherwise
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
        is_cpu_intensive += 1 if us > wa else -1
        io_stats.append(wa)

        lines = lines[-2:]
        assert lines[0].split()[8] == "%CPU"
        cpu_percentage = float(lines[1].split()[8])
        cpu_stat = round(float(cpu_percentage / utils.get_cpu_count()), 1)
        cpu_stats.append(cpu_stat)
        time.sleep(0.1)
    timer.cancel()
    throughput = time.time_ns() - time_start

    cpu_avg = round(float(numpy.mean(cpu_stats)), 1)
    io_avg = round(float(numpy.mean(io_stats)), 1)
    if process.returncode != 0:
        error = process.stderr.read().decode() if process.stderr is not None else ""
        return (
            io_avg < 10,
            cpu_avg,
            io_avg,
            throughput,
            (process.returncode, error),
        )

    return (
        io_avg < 10,
        cpu_avg,
        io_avg,
        throughput,
        None,
    )


def run_benchmark_groups(
    gc: str,
    heap_size: str,
    iterations: int,
    jdk: str,
    timeout: int,
    benchmark_groups: Optional[list[BENCHMARK_GROUP]],
) -> list[BenchmarkResult]:
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
        list[BenchmarkResult]
    """
    benchmark_results: list[BenchmarkResult] = []
    groups: list[BENCHMARK_GROUP] = benchmark_groups or [*BENCHMARK_GROUP]

    for benchmark_group in groups:
        benchmarks = _get_benchmarks(benchmark_group)

        for benchmark in benchmarks:
            command = _get_benchmark_command(
                benchmark_group, benchmark, gc, heap_size, iterations
            )

            print(
                f"[{benchmark_group.value}]: Running benchmark {benchmark} with:\n\tGC: {gc}\n\tHeap Size: {heap_size}\n\tIterations: {iterations=}"
            )

            (cpu_intensive, average_cpu, average_io, throughput, error) = run_benchmark(
                command, timeout
            )

            print(f"{cpu_intensive=} {average_cpu=} {average_io=} and {throughput=}")
            if error is None:
                result = BenchmarkResult.build_benchmark_result(
                    gc,
                    benchmark_group.value,
                    benchmark,
                    heap_size,
                    cpu_intensive,
                    average_cpu,
                    average_io,
                    throughput,
                    jdk,
                )
            else:
                result = BenchmarkResult.build_benchmark_error(
                    gc,
                    benchmark_group.value,
                    benchmark,
                    heap_size,
                    cpu_intensive,
                    average_cpu,
                    average_io,
                    jdk,
                    error[0],
                    error[1],
                )
            result.save_to_json()
            benchmark_results.append(result)
    return benchmark_results


def run_benchmarks(
    iterations: int,
    jdk: str,
    garbage_collectors: list[str],
    skip_benchmarks: bool,
    benchmark_groups: list[BENCHMARK_GROUP],
) -> list[GarbageCollectorResult]:
    benchmark_results: dict[str, dict[str, list[BenchmarkResult]]] = {}
    heap_sizes: list[str] = utils.get_heap_sizes()

    failed_benchmarks: dict[str, dict[str, list[tuple[str, str]]]] = {}

    for gc in garbage_collectors:
        benchmark_results[gc] = {}
        for heap_size in heap_sizes:
            if heap_size not in benchmark_results[gc]:
                benchmark_results[gc][heap_size] = []

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
                    if result.heap_size not in failed_benchmarks:
                        failed_benchmarks[result.heap_size] = {}
                    if result.benchmark_name not in failed_benchmarks[result.heap_size]:
                        failed_benchmarks[result.heap_size][result.benchmark_name] = []

                    failed_benchmarks[result.heap_size][result.benchmark_name].append(
                        (
                            result.garbage_collector,
                            result.error,  # type: ignore
                        )
                    )

    gc_results: list[GarbageCollectorResult] = []
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
            # print(benchmark_results[gc][heap_size])

        if len(benchmark_results[gc]) > 0:
            gc_result = GarbageCollectorResult.build_garbage_collector_result(
                benchmark_results[gc]
            )
            gc_result.save_to_json()
            gc_results.append(gc_result)
        else:
            f"Garbage Collector {gc} doesn't have successfull benchmarks."

    if len(failed_benchmarks) > 0:
        error_report = _ErrorReport(jdk, failed_benchmarks)
        error_report.save_to_json()
    return gc_results
