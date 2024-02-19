#!/usr/bin/python3

import os
import argparse
import subprocess
import time

import numpy

import utils
from model import BenchmarkResult, GarbageCollectorResult, Stats


def run_benchmark(
    benchmark_command: str,
) -> Stats:
    process = subprocess.Popen(benchmark_command)
    time_start = time.time_ns()
    pid = process.pid
    cpu_stats = []

    # TODO: see better polling method than sleeping 1 seconds for throughput
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
        time.sleep(1)

    return (round(float(numpy.mean(cpu_stats)), 1), time.time_ns() - time_start)


def run_renaissance(gc: str, iterations: int) -> list[BenchmarkResult]:
    benchmark_results: list[BenchmarkResult] = []
    benchmark_group = "Renaissance"

    result = subprocess.run(
        ["java", "-jar", utils.get_benchmark_jar_path(benchmark_group), "--raw-list"],
        capture_output=True,
        text=True,
    )
    renaissance_benchmarks = result.stdout.splitlines()

    # for benchmark in renaissance_benchmarks:
    for benchmark in renaissance_benchmarks[0:2]:
        command = [
            "java",
            f"-XX:+Use{gc}GC",
            f"-Xlog:gc*,safepoint:file={utils.get_benchmark_log_path(gc, benchmark_group, benchmark)}::filecount=0",
            "-jar",
            utils.get_benchmark_jar_path(benchmark_group),
            benchmark,
            "-r",
            f"{iterations}",
            "--no-forced-gc",
        ]

        print(f"Running benchmark {benchmark} with GC: {gc} and {iterations=}")
        (average_cpu, throughput) = run_benchmark(command)

        print(f"{average_cpu=} and {throughput=}")
        result = BenchmarkResult.build_benchmark_result(
            gc,
            benchmark_group,
            benchmark,
            utils.is_cpu_intensive(average_cpu),
            throughput,
        )
        result.save_to_json()
        benchmark_results.append(result)

    return benchmark_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute throughput and average pause time for benchmarks"
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
        help="Number of iterations to run benchmarks. Increase this number to achieve more reliable metrics.",
    )
    parser.print_help()

    args = parser.parse_args(argv)
    skip_benchmarks = args.skip_benchmarks
    iterations = args.iterations

    for gc in ["G1"]:
        benchmark_results: list[BenchmarkResult] = []
        if skip_benchmarks:
            benchmark_results = utils.load_benchmark_results()
        else:
            benchmark_results.extend(run_renaissance(gc, iterations))

        total_gc_pauses = 0
        total_gc_pause_time = 0
        p90_gc_pause_time = []
        gc_throughput = []

        for result in benchmark_results:
            total_gc_pauses += result.number_of_pauses
            total_gc_pause_time += result.total_pause_time
            p90_gc_pause_time.append(result.p90_pause_time)
            gc_throughput.append(result.throughput)

        gc_result = GarbageCollectorResult(
            gc,
            total_gc_pauses,
            total_gc_pause_time,
            round(total_gc_pause_time / total_gc_pauses, 2),
            round(numpy.mean(p90_gc_pause_time), 2),
            round(numpy.mean(gc_throughput), 2),
        )
        print(f"{gc_result}")
        gc_result.save_to_json()
    return 0


if __name__ == "__main__":
    exit(main())
