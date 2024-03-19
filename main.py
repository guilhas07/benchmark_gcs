#!/usr/bin/env python

import argparse
import re
import subprocess
import time
from threading import Timer
from typing import Optional

import numpy

import utils
from model import (
    BenchmarkResult,
    GarbageCollectorResult,
    StatsMatrix,
    _ErrorReport,
)

Benchmark_Stats = tuple[bool, float, float, int, Optional[tuple[int, str]]]
"""(cpu_intensive, average_cpu_usage_percentage, average_io_percentage, throughput, error)
cpu_intensive: bool -> True if application is cpu_bound, False otherwise
average_cpu_usage_percentage: float -> average cpu usage percentage of the benchmark
average_io_percentage: float -> average IO percentage of the benchmark
throughput: int -> Time in nanoseconds. Equivalent to program execution time
error: Optional[tuple[int, str]] -> Application return code and error message when application fails"""


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
    if process.returncode != 0:
        error = process.stderr.read().decode() if process.stderr is not None else ""
        return (
            is_cpu_intensive > 0,
            round(float(numpy.mean(cpu_stats)), 1),
            round(float(numpy.mean(io_stats)), 1),
            throughput,
            (process.returncode, error),
        )

    return (
        is_cpu_intensive > 0,
        round(float(numpy.mean(cpu_stats)), 1),
        round(float(numpy.mean(io_stats)), 1),
        throughput,
        None,
    )


def run_renaissance(
    gc: str, jdk: str, iterations: int, heap_size: str
) -> list[BenchmarkResult]:
    benchmark_results: list[BenchmarkResult] = []
    benchmark_group = "Renaissance"

    result = subprocess.run(
        ["java", "-jar", utils.get_benchmark_jar_path(benchmark_group), "--raw-list"],
        capture_output=True,
        text=True,
    )
    assert result.stderr == "", result.stderr
    renaissance_benchmarks = result.stdout.splitlines()

    for benchmark in renaissance_benchmarks:
        command = [
            "java",
            f"-XX:+Use{gc}GC",
            f"-Xms{heap_size}m",
            f"-Xmx{heap_size}m",
            f"-Xlog:gc*,safepoint:file={utils.get_benchmark_log_path(gc, benchmark_group, benchmark, heap_size)}::filecount=0",
            "-jar",
            utils.get_benchmark_jar_path(benchmark_group),
            benchmark,
            "-r",
            f"{iterations}",
            "--no-forced-gc",
        ]

        print(
            f"Running benchmark {benchmark} with:\n\tGC: {gc}\n\tHeap Size: {heap_size}\n\tIterations: {iterations=}"
        )

        (cpu_intensive, average_cpu, average_io, throughput, error) = run_benchmark(
            command, iterations * 60
        )

        print(f"{cpu_intensive=} {average_cpu=} {average_io=} and {throughput=}")
        if error is None:
            result = BenchmarkResult.build_benchmark_result(
                gc,
                benchmark_group,
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
                benchmark_group,
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
    iterations: int, jdk: str, garbage_collectors: list[str], skip_benchmarks=False
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
                        run_renaissance(gc, jdk, iterations, heap_size)
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

    error_report = _ErrorReport(jdk, failed_benchmarks)
    error_report.save_to_json()
    return gc_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute throughput and average pause time for benchmarks"
    )
    parser.add_argument(
        "-c",
        "--clean",
        dest="clean",
        action="store_true",
        help="Clean the benchmark stats and log paths",
    )
    parser.add_argument(
        "-s",
        "--skip_benchmarks",
        dest="skip_benchmarks",
        action="store_true",
        help="""Skip the benchmarks and compute the matrix with previously obtained garbage collector results.
        Specify the java jdk of the garbage collector results if your current java jdk version is different than the one of the results'.""",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        dest="iterations",
        default=10,
        type=int,
        help="Number of iterations to run benchmarks. Increase this number to achieve more reliable metrics.",
    )

    parser.add_argument(
        "-j",
        "--jdk",
        dest="jdk",
        choices=utils.get_supported_jdks(),
        help="Specify the java jdk version when you wish to skip the benchmarks and only calculate the matrix.",
    )

    args = parser.parse_args(argv)
    parser.print_help()
    skip_benchmarks = args.skip_benchmarks
    iterations = args.iterations
    clean = args.clean
    jdk = args.jdk

    if clean:
        utils.clean_logs_and_stats()
        if skip_benchmarks:
            print("Cleaned and skipped benchmarks")
            return 0

    gc_results = []
    garbage_collectors = []
    if skip_benchmarks:
        assert (
            jdk is not None
        ), "Please specify a java version in order to skip the benchmarks"
        _, garbage_collectors = utils.get_available_gcs(jdk)

        # gc_results = utils.load_garbage_collector_results(jdk)
        # TODO: support not computing stats again
    else:
        jdk, garbage_collectors = utils.get_available_gcs()

    assert (
        jdk is not None and garbage_collectors is not None
    ), "Current jdk is not supported"

    gc_results = run_benchmarks(iterations, jdk, garbage_collectors, skip_benchmarks)

    if len(gc_results) == 0:
        print("No GarbageCollector had successfull benchmarks")
        return 0

    matrix = StatsMatrix.build_stats_matrix(gc_results, "G1")
    matrix.save_to_json(jdk)
    return 0


if __name__ == "__main__":
    exit(main())
