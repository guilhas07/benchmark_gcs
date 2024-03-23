#!/usr/bin/env python

import argparse

import utils
import benchmark
from model import (
    StatsMatrix,
)


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

    parser.add_argument(
        "-b",
        "--benchmarks",
        dest="benchmarks",
        choices=[group.value for group in benchmark.BENCHMARK_GROUP],
        help="Specify the group of benchmarks to run",
        nargs="+",
    )

    args = parser.parse_args(argv)
    parser.print_help()
    skip_benchmarks = args.skip_benchmarks
    iterations = args.iterations
    clean = args.clean
    jdk = args.jdk
    benchmarks = args.benchmarks and [
        benchmark.BENCHMARK_GROUP(el) for el in args.benchmarks
    ]

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

    gc_results = benchmark.run_benchmarks(
        iterations, jdk, garbage_collectors, skip_benchmarks, benchmarks
    )

    if len(gc_results) == 0:
        print("No GarbageCollector had successfull benchmarks")
        return 0

    matrix = StatsMatrix.build_stats_matrix(gc_results, "G1")
    matrix.save_to_json(jdk)
    return 0


if __name__ == "__main__":
    exit(main())
