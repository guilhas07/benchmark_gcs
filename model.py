from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy

import utils

Benchmark_Stats = tuple[float, int]
"""(average_cpu_percentage, throughput)
average_cpu_percentage: float -> average benchmark cpu percentage
throughput: int -> Time in nanoseconds. Equivalent to program execution time"""


@dataclass
class GarbageCollectorStats:
    heap_size: int
    number_of_pauses: int
    total_pause_time: int
    avg_pause_time: float
    p90_avg_pause_time: float
    avg_throughput: float

    @staticmethod
    def build_gc_stats(
        heap_size: int, benchmark_results: list[BenchmarkResult]
    ) -> GarbageCollectorStats:
        """
        Args:
            heap_size: int -> size of the heap used for the benchmark
            benchmark_results: list[BenchmarkResult] -> Valid benchmark results with atleast one element. Meaning
            only successfull "BenchmarkResult"s.
            You can check the validity of the benchmark with benchmark_result.is_successfull()

        Returns:
            stats: GarbageCollectorStats
        """
        total_gc_pauses = 0
        total_gc_pause_time = 0
        p90_gc_pause_time = []
        gc_throughput = []

        assert len(benchmark_results) > 0
        assert all(el.is_successfull() for el in benchmark_results)

        for result in benchmark_results:
            total_gc_pauses += result.number_of_pauses
            total_gc_pause_time += result.total_pause_time
            p90_gc_pause_time.append(result.p90_pause_time)
            gc_throughput.append(result.throughput)

        return GarbageCollectorStats(
            heap_size,
            total_gc_pauses,
            total_gc_pause_time,
            round(total_gc_pause_time / total_gc_pauses, 2),
            round(numpy.mean(p90_gc_pause_time), 2),
            round(numpy.mean(gc_throughput), 2),
        )


@dataclass
class GarbageCollectorResult:
    garbage_collector: str
    stats: list[GarbageCollectorStats]

    def save_to_json(self):
        with open(
            utils.get_gc_global_stats_path(self.garbage_collector),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))


@dataclass
class BenchmarkResult:
    garbage_collector: str
    benchmark_group: str
    benchmark_name: str
    heap_size: int
    success: tuple[bool, str]
    cpu_intensive: bool
    number_of_pauses: int
    total_pause_time: int
    avg_pause_time: float
    pauses_per_category: dict[str, int]
    total_pause_time_per_category: dict[str, int]
    avg_pause_time_per_category: dict[str, float]
    p90_pause_time: float
    throughput: int

    @staticmethod
    def _map_error_code(error_code: int) -> str:
        error_codes = {-9: "Process was killed due to timeout", -1: "Out of memory"}
        if error_code in error_codes:
            return error_codes[error_code]
        return "Error code not found"

    @staticmethod
    def load_from_json(file_path: str) -> BenchmarkResult:
        with open(file_path) as f:
            return BenchmarkResult(**json.loads(f.read()))

    @staticmethod
    def build_benchmark_error(
        gc, benchmark_group, benchmark, heap_size, error_code: int
    ) -> BenchmarkResult:
        return BenchmarkResult(
            gc,
            benchmark_group,
            benchmark,
            heap_size,
            (
                False,
                f"Return code {error_code}: {BenchmarkResult._map_error_code(error_code)}",
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def build_benchmark_result(
        gc: str,
        benchmark_group: str,
        benchmark_name: str,
        heap_size: int,
        cpu_intensive: bool,
        throughput: int,
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

        p90_pause_time = round(numpy.percentile(pause_times, 90), 2)
        avg_pause_time = round(total_pause_time / number_of_pauses, 2)

        for category, pause_time in total_pause_time_per_category.items():
            avg_pause_time_per_category[category] = round(
                pause_time / pauses_per_category[category], 2
            )

        return BenchmarkResult(
            gc,
            benchmark_group,
            benchmark_name,
            heap_size,
            (True, None),
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
        return self.success[0]

    def save_to_json(self):
        with open(
            utils.get_benchmark_stats_path(
                self.garbage_collector,
                self.benchmark_group,
                self.benchmark_name,
                self.heap_size,
                self.success,
            ),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))
