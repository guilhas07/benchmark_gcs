#!/usr/bin/python3

import argparse
import subprocess
import time
from threading import Timer

import numpy

import utils
from model import (
    Benchmark_Stats,
    BenchmarkResult,
    GarbageCollectorResult,
    GarbageCollectorStats,
)


def run_benchmark(benchmark_command: str, timeout: int) -> Benchmark_Stats:
    """
    Args:
        benchmark_command: str -> command to run benchmark
        timeout: int -> value in seconds to interrupt benchmark
    Returns:
        BenchmarkStats
    """

    process = subprocess.Popen(benchmark_command)
    timer = Timer(1, process.kill)
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
        cpu_stat = round(float(cpu_percentage / utils.CPU_COUNT), 1)
        print(f"{cpu_stat=}")
        cpu_stats.append(cpu_stat)
        time.sleep(0.1)
    timer.cancel()
    if process.returncode != 0:
        raise Exception(process.returncode)

    return (round(float(numpy.mean(cpu_stats)), 1), time.time_ns() - time_start)


def run_renaissance(gc: str, iterations: int, heap_size: int) -> list[BenchmarkResult]:
    benchmark_results: list[BenchmarkResult] = []
    benchmark_group = "Renaissance"

    result = subprocess.run(
        ["java", "-jar", utils.get_benchmark_jar_path(benchmark_group), "--raw-list"],
        capture_output=True,
        text=True,
    )
    renaissance_benchmarks = result.stdout.splitlines()

    for benchmark in renaissance_benchmarks:
        # for benchmark in renaissance_benchmarks[0:2]:
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
                utils.is_cpu_intensive(average_cpu),
                throughput,
            )
        except Exception as e:
            result = BenchmarkResult.build_benchmark_error(
                gc, benchmark_group, benchmark, heap_size, e.args[0]
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

    benchmark_results: dict[tuple[str, str], list[BenchmarkResult]] = {}
    gcs: list[str] = utils.get_available_gcs()
    heap_sizes: list[str] = utils.get_heap_sizes()

    failed_benchmarks: dict[tuple[str, str], list[str]] = {}
    # run the benchmakrs
    for gc in gcs:
        # gc_stats: GarbageCollectorStats = []
        # for heap_size in utils.get_heap_sizes():
        for heap_size in heap_sizes[0:2]:
            if skip_benchmarks:
                benchmark_results[(gc, heap_size)] = utils.load_benchmark_results(
                    gc, heap_size
                )
            else:
                benchmark_results[(gc, heap_size)] = []
                benchmark_results[(gc, heap_size)].extend(
                    run_renaissance(gc, iterations, heap_size)
                )

            for result in benchmark_results[(gc, heap_size)]:
                if not result.is_successfull():
                    print(
                        f"Benchmark {result.benchmark_name} failed with {gc=} and {heap_size=}."
                    )
                    if (
                        result.benchmark_name,
                        result.heap_size,
                    ) not in failed_benchmarks:
                        failed_benchmarks[
                            (result.benchmark_name, result.heap_size)
                        ] = []

                    failed_benchmarks[(result.benchmark_name, result.heap_size)].append(
                        gc
                    )

            # gc_stats.append(
            #     GarbageCollectorStats.build_gc_stats(heap_size, benchmark_results)
            # )

    for (name, heap_size), value in failed_benchmarks.items():
        print(f"Benchmark {name} failed for heap size {heap_size} for gcs: {value}")

    # compute stats
    gc_stats: dict[str, list[GarbageCollectorStats]] = {}
    for (gc, heap_size), results in benchmark_results.items():
        valid_results = [
            el
            for el in results
            if (el.benchmark_name, el.heap_size) not in failed_benchmarks
        ]

        if len(valid_results) == 0:
            continue

        if gc not in gc_stats:
            gc_stats[gc] = []
        gc_stats[gc].append(
            GarbageCollectorStats.build_gc_stats(heap_size, valid_results)
        )

    for gc, stats in gc_stats.items():
        gc_result = GarbageCollectorResult(gc, stats)
        print(f"{gc_result}")
        gc_result.save_to_json()

    return 0


if __name__ == "__main__":
    # TODO: define custom exception
    exit(main())
