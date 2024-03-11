from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

import numpy

import utils

Benchmark_Stats = tuple[float, int]
"""(average_cpu_percentage, throughput)
average_cpu_percentage: float -> average benchmark cpu percentage
throughput: int -> Time in nanoseconds. Equivalent to program execution time"""


@dataclass
class StatsMatrix:
    @dataclass
    class GCScore:
        throughput: float
        pause_time: float

    cpu_intensive_matrix: dict[
        str, dict[str, GCScore]
    ]  # {heap_size: { gc : {throughput, pause_time }}}
    non_cpu_intensive_matrix: dict[
        str, dict[str, GCScore]
    ]  # {heap_size: { gc : {throughput, pause_time }}}
    garbage_collectors: list[str]
    benchmarks: dict[str, dict[str, list[str]]]

    @staticmethod
    def build_stats_matrix(
        gc_results: list[GarbageCollectorResult], default_gc: str
    ) -> StatsMatrix:
        def compute_matrix(
            default_stats: list[GarbageCollectorResult.GarbageCollectorStats],
            gc_results: list[GarbageCollectorResult],
            cpu_intensive: bool,
        ):
            matrix: dict[str, dict[str, StatsMatrix.GCScore]] = {}
            for id, stat in enumerate(default_stats):
                normalization_throughtput_factor = stat.avg_throughput
                normalization_p90_pause_time_factor = stat.p90_avg_pause_time
                if stat.heap_size not in matrix:
                    matrix[stat.heap_size] = {}

                for result in gc_results:
                    gc_stat = (
                        result.cpu_intensive_stats[id]
                        if cpu_intensive
                        else result.non_cpu_intensive_stats[id]
                    )
                    matrix[stat.heap_size][
                        result.garbage_collector
                    ] = StatsMatrix.GCScore(
                        round(
                            gc_stat.avg_throughput / normalization_throughtput_factor,
                            2,
                        ),
                        round(
                            gc_stat.p90_avg_pause_time
                            / normalization_p90_pause_time_factor,
                            2,
                        ),
                    )

            return matrix

        assert (
            len(gc_results) > 0
        ), "gc_results should be a list with  one or more elements"

        default_gc_element: GarbageCollectorResult | None = None

        garbage_collectors = set()
        for result in gc_results:
            assert (
                result.garbage_collector not in garbage_collectors
            ), "Results should be of distinct garbage collectors."
            garbage_collectors.add(result.garbage_collector)
            if result.garbage_collector == default_gc:
                default_gc_element = result

        assert default_gc_element is not None, "default_gc was not found in gc_results"

        assert all(
            len(default_gc_element.cpu_intensive_stats) == len(el.cpu_intensive_stats)
            for el in gc_results
        ), "All gc_results should have cpu_intensive_stats with the same length"

        assert all(
            len(default_gc_element.non_cpu_intensive_stats)
            == len(el.non_cpu_intensive_stats)
            for el in gc_results
        ), "All gc_results should have non_cpu_intensive_stats with the same length"

        benchmarks: dict[str, dict[str, list[str]]] = {}

        for id, stat in enumerate(default_gc_element.cpu_intensive_stats):
            assert all(
                stat.heap_size == el.cpu_intensive_stats[id].heap_size
                for el in gc_results
            ), "All gc_results should have cpu_intensive stats for the same heap_sizes"

            assert all(
                stat.benchmarks == el.cpu_intensive_stats[id].benchmarks
                for el in gc_results
            ), "All gc_results should have cpu_intensive stats for the same benchmarks"
            # TODO: benchmarks can be cpu_intensive for on GC and not for another ??
            benchmarks["cpu_intensive"] = {stat.heap_size: stat.benchmarks}

        for id, stat in enumerate(default_gc_element.non_cpu_intensive_stats):
            assert all(
                stat.heap_size == el.non_cpu_intensive_stats[id].heap_size
                for el in gc_results
            ), "All gc_results should have non_cpu_intensive stats for the same heap_sizes"
            assert all(
                stat.benchmarks == el.non_cpu_intensive_stats[id].benchmarks
                for el in gc_results
            ), "All gc_results should have non_cpu_intensive stats for the same benchmarks"
            benchmarks["non_cpu_intensive"] = {stat.heap_size: stat.benchmarks}

        cpu_intensive_matrix = compute_matrix(
            default_gc_element.cpu_intensive_stats, gc_results, True
        )
        non_cpu_intensive_matrix = compute_matrix(
            default_gc_element.non_cpu_intensive_stats, gc_results, False
        )

        return StatsMatrix(
            cpu_intensive_matrix,
            non_cpu_intensive_matrix,
            list(garbage_collectors),
            benchmarks,
        )

    def save_to_json(self, jdk: str):
        date = datetime.datetime.now()
        day = date.day
        month = date.month
        hour = date.hour
        minute = date.minute

        with open(
            utils.get_matrix_path(jdk, f"{day}_{month}_{hour}_{minute}"),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))


@dataclass
class GarbageCollectorResult:
    garbage_collector: str
    jdk: str
    cpu_intensive_stats: list[GarbageCollectorStats]
    non_cpu_intensive_stats: list[GarbageCollectorStats]

    @dataclass
    class GarbageCollectorStats:
        heap_size: str
        number_of_pauses: int
        total_pause_time: int
        avg_pause_time: float
        p90_avg_pause_time: float
        avg_throughput: float
        benchmarks: list[str]

    @staticmethod
    def _build_gc_stats(
        heap_size: str,
        benchmark_results: list[BenchmarkResult],
    ) -> GarbageCollectorResult.GarbageCollectorStats:
        """
        Args:
            heap_size: int -> size of the heap used for the list of benchmark results.
            benchmark_results: list[BenchmarkResult] -> Valid benchmark results (meaning only successfull benchmarks), with atleast one element.
            All benchmark results should have an heap_size equal to the provided heap_size and should be from the same garbage collector.
            You can check the validity of the benchmark with benchmark_result.is_successfull().

        Returns:
            stats: GarbageCollectorStats
        """
        total_gc_pauses: int = 0
        total_gc_pause_time: int = 0
        p90_gc_pause_time = []
        gc_throughput = []
        benchmarks = []

        for result in benchmark_results:
            total_gc_pauses += result.number_of_pauses
            total_gc_pause_time += result.total_pause_time
            p90_gc_pause_time.append(result.p90_pause_time)
            gc_throughput.append(result.throughput)

            assert (
                result.benchmark_name not in benchmarks
            ), "There shouldn't be repeated benchmarks"
            benchmarks.append(result.benchmark_name)

        benchmarks.sort()
        return GarbageCollectorResult.GarbageCollectorStats(
            heap_size,
            total_gc_pauses,
            total_gc_pause_time,
            round(total_gc_pause_time / total_gc_pauses, 2),
            float(round(numpy.mean(p90_gc_pause_time), 2)),
            float(round(numpy.mean(gc_throughput), 2)),
            benchmarks,
        )

    @staticmethod
    def build_garbage_collector_result(
        benchmarks_results: dict[str, list[BenchmarkResult]],
    ) -> GarbageCollectorResult:
        """Builds a garbage collector result, taking into account every successfull benchmark run for every heap_size.

        Args:
            benchmarks_results: Dictionary with `heap_size` as key and value as the list of `BenchmarkResult`s for that `heap_size`.
                                NOTE: Every value should have one or more elements.

        Returns:
            A `GarbageCollectorResult`
        """
        cpu_intensive_stats = []
        non_cpu_intensive_stats = []

        assert (
            len(benchmarks_results) > 0
        ), "benchmark_results should have one or more keys"

        gc = ""
        jdk = ""
        for key, value in benchmarks_results.items():
            assert (
                len(value) >= 0
            ), "List of benchmark results should have one element or more"

            if gc == "" and jdk == "":
                gc = value[0].garbage_collector
                jdk = value[0].jdk

            assert all(
                el.is_successfull() for el in value
            ), "All benchmarks should be successfull"

            assert all(
                el.heap_size == key for el in value
            ), f"All benchmark results should have the same heap size {key}"

            assert all(
                el.is_successfull() for el in value
            ), "All benchmark results should be successfull"

            assert all(
                el.garbage_collector == gc for el in value
            ), "All benchmark results should be from the same garbage collector"

            assert all(
                el.jdk == jdk for el in value
            ), "All benchmark results should be from the same runtime"

        for heap_size in benchmarks_results:
            cpu_intensive_benchmarks = [
                el for el in benchmarks_results[heap_size] if el.cpu_intensive[0]
            ]
            non_cpu_intensive_benchmarks = [
                el for el in benchmarks_results[heap_size] if not el.cpu_intensive[0]
            ]

            if len(cpu_intensive_benchmarks) > 0:
                cpu_intensive_stats.append(
                    GarbageCollectorResult._build_gc_stats(
                        heap_size, cpu_intensive_benchmarks
                    )
                )

            if len(non_cpu_intensive_benchmarks) > 0:
                non_cpu_intensive_stats.append(
                    GarbageCollectorResult._build_gc_stats(
                        heap_size, non_cpu_intensive_benchmarks
                    )
                )

        return GarbageCollectorResult(
            gc, jdk, cpu_intensive_stats, non_cpu_intensive_stats
        )

    def save_to_json(self):
        with open(
            utils.get_gc_stats_path(self.garbage_collector, self.jdk),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))


@dataclass
class BenchmarkResult:
    garbage_collector: str
    jdk: str
    benchmark_group: str
    benchmark_name: str
    heap_size: str
    error: Optional[str]
    cpu_intensive: tuple[bool, float]
    number_of_pauses: int
    total_pause_time: int
    avg_pause_time: float
    pauses_per_category: dict[str, int]
    total_pause_time_per_category: dict[str, int]
    avg_pause_time_per_category: dict[str, float]
    p90_pause_time: float
    throughput: int

    @staticmethod
    def load_from_json(file_path: str) -> BenchmarkResult:
        with open(file_path) as f:
            return BenchmarkResult(**json.loads(f.read()))

    @staticmethod
    def build_benchmark_error(
        gc: str,
        benchmark_group: str,
        benchmark: str,
        heap_size: str,
        jdk: str,
        error_code: int,
        error_message: str,
    ) -> BenchmarkResult:
        def error_code_message(error_code: int, error_message: str) -> str:
            error_codes = {-9: "Process was killed due to timeout", -1: "Out of memory"}
            if error_code in error_codes:
                return f"Return code {error_code}: {error_codes[error_code]}\n{error_message}"
            return f"Error code not found: {error_code}\n{error_message}"

        return BenchmarkResult(
            gc,
            jdk,
            benchmark_group,
            benchmark,
            heap_size,
            # (
            #     False,
            error_code_message(error_code, error_message),
            # ),
            (False, -1),
            0,
            0,
            0,
            {},
            {},
            {},
            0,
            0,
        )

    @staticmethod
    def build_benchmark_result(
        gc: str,
        benchmark_group: str,
        benchmark_name: str,
        heap_size: str,
        cpu_intensive: tuple[bool, float],
        throughput: int,
        jdk: str,
    ) -> BenchmarkResult:
        log_file = utils.get_benchmark_log_path(
            gc, benchmark_group, benchmark_name, heap_size
        )
        number_of_pauses = 0

        total_pause_time = 0
        avg_pause_time = 0.0
        pauses_per_category = {}
        total_pause_time_per_category = {}
        avg_pause_time_per_category = {}
        pause_times = []

        with open(log_file) as f:
            for line in f:
                if "safepoint" in line:
                    # use () to group
                    pause_category = re.findall('Safepoint "(.*)"', line)[0]
                    pause_time = int(re.findall("Total: (\\d+) ns", line)[0])
                    pause_times.append(pause_time)

                    if pause_category not in pauses_per_category:
                        pauses_per_category[pause_category] = 0
                        total_pause_time_per_category[pause_category] = 0

                    pauses_per_category[pause_category] += 1
                    total_pause_time_per_category[pause_category] += pause_time
                    total_pause_time += pause_time
                    number_of_pauses += 1

        p90_pause_time = float(round(numpy.percentile(pause_times, 90), 2))
        avg_pause_time = float(round(total_pause_time / number_of_pauses, 2))

        for category, pause_time in total_pause_time_per_category.items():
            avg_pause_time_per_category[category] = round(
                pause_time / pauses_per_category[category], 2
            )

        return BenchmarkResult(
            gc,
            jdk,
            benchmark_group,
            benchmark_name,
            heap_size,
            None,
            cpu_intensive,
            number_of_pauses,
            total_pause_time,
            avg_pause_time,
            pauses_per_category,
            total_pause_time_per_category,
            avg_pause_time_per_category,
            p90_pause_time,
            throughput,
        )

    def is_successfull(self):
        return self.error is None

    def save_to_json(self):
        with open(
            utils.get_benchmark_stats_path(
                self.garbage_collector,
                self.benchmark_group,
                self.benchmark_name,
                self.heap_size,
                self.jdk,
                self.error,
            ),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))


@dataclass
class _ErrorReport:
    jdk: str
    failed_benchmarks: dict[str, dict[str, list[tuple[str, str]]]]

    def save_to_json(self):
        date = datetime.datetime.now()
        day = date.day
        month = date.month
        hour = date.hour
        minute = date.minute
        with open(
            utils.get_error_report_path(self.jdk, f"{day}_{month}_{hour}_{minute}"),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))
