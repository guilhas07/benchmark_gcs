from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy

import utils

Stats = tuple[float, int]
"""(average_cpu_percentage, throughput)
average_cpu_percentage: float -> average benchmark cpu percentage
throughput: int -> Time in nanoseconds. Equivalent to program execution time"""


@dataclass
class GarbageCollectorResult:
    garbage_collector: str
    number_of_pauses: int
    total_pause_time: int
    avg_pause_time: float
    p90_avg_pause_time: float
    avg_throughput: float

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
    def load_from_json(file_path: str) -> BenchmarkResult:
        with open(file_path) as f:
            return BenchmarkResult(**json.loads(f.read()))

    @staticmethod
    def build_benchmark_result(
        gc: str,
        benchmark_group: str,
        benchmark_name: str,
        cpu_intensive: bool,
        throughput: int,
    ) -> BenchmarkResult:
        log_file = utils.get_benchmark_log_path(gc, benchmark_group, benchmark_name)
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

    def save_to_json(self):
        with open(
            utils.get_benchmark_stats_path(
                self.garbage_collector, self.benchmark_group, self.benchmark_name
            ),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))
