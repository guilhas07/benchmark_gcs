#!/usr/bin/env python

import argparse
import subprocess
import time
from threading import Timer

import numpy

import utils
from model import (
    Benchmark_Stats,
    BenchmarkResult,
    _ErrorReport,
    GarbageCollectorResult,
    StatsMatrix,
)


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
    # timer = Timer(timeout, process.kill)
    timer = Timer(1, process.kill)  # TODO: remove this
    time_start = time.time_ns()
    pid = process.pid
    cpu_stats = []

    # TODO: see better polling method than sleeping 1 seconds for throughput
    timer.start()
    while process.poll() is None:
        # subprocess.run with capture_output doesn't seem to capture the whole output when
        # using top with -1 flag
        p = subprocess.run(
            ["top", "-bn", "1", "-p", f"{pid}"], capture_output=True, text=True
        )
        lines = p.stdout.splitlines()[-2:]
        assert lines[0].split()[8] == "%CPU"
        cpu_percentage = float(lines[1].split()[8])
        cpu_stat = round(float(cpu_percentage / utils.get_cpu_count()), 1)
        # print(f"{cpu_stat=}")
        cpu_stats.append(cpu_stat)
        time.sleep(1)
    timer.cancel()
    error = process.stderr.read().decode() if process.stderr is not None else ""
    if process.returncode != 0:
        raise Exception(process.returncode, error)

    return (round(float(numpy.mean(cpu_stats)), 1), time.time_ns() - time_start)


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
        try:
            (average_cpu, throughput) = run_benchmark(command, iterations * 60)

            print(f"{average_cpu=} and {throughput=}")
            result = BenchmarkResult.build_benchmark_result(
                gc,
                benchmark_group,
                benchmark,
                heap_size,
                (utils.is_cpu_intensive(average_cpu), average_cpu),
                throughput,
                jdk,
            )
        except Exception as e:
            result = BenchmarkResult.build_benchmark_error(
                gc, benchmark_group, benchmark, heap_size, jdk, e.args[0], e.args[1]
            )
        result.save_to_json()
        benchmark_results.append(result)

    return benchmark_results


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
        help="Skip the benchmarks",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        dest="iterations",
        default=10,
        type=int,
        help="Number of iterations to run benchmarks. Increase this number to achieve more reliable metrics.",
    )

    args = parser.parse_args(argv)
    parser.print_help()
    skip_benchmarks = args.skip_benchmarks
    iterations = args.iterations
    clean = args.clean

    if clean:
        utils.clean_logs_and_stats()
        if skip_benchmarks:
            print("Cleaned and skipped benchmarks")
            return 0

    jdk, gcs = utils.get_available_gcs()
    assert jdk is not None and gcs is not None, "Current jdk is not supported"

    benchmark_results: dict[str, dict[str, list[BenchmarkResult]]] = {}
    heap_sizes: list[str] = utils.get_heap_sizes()
    # heap_sizes.reverse()  # NOTE: start with higher heap_sizes
    heap_sizes = heap_sizes[0:1]  # TODO: REMOVE THIS
    # failed_benchmarks: dict[tuple[str, str], list[str]] = {}

    failed_benchmarks: dict[str, dict[str, list[tuple[str, str]]]] = {}

    # run the benchmakrs
    for gc in gcs:
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

                    assert (
                        result.error is not None
                    ), "Failed benchmark should have a str error message"
                    failed_benchmarks[result.heap_size][result.benchmark_name].append(
                        (
                            result.garbage_collector,
                            result.error,
                        )
                    )

    gc_results = []
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
            print(benchmark_results[gc][heap_size])

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

    if len(gc_results) > 0:
        matrix = StatsMatrix.build_stats_matrix(gc_results, "G1")
        matrix.save_to_json(jdk)
    else:
        "No GarbageCollector had successfull benchmarks"
    return 0


if __name__ == "__main__":
    # TODO: define custom exception
    exit(main())
