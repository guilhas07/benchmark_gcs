import glob
import os

from model import BenchmarkResult

CPU_COUNT = os.cpu_count()
_CPU_THRESHOLD = 60

_dir = os.path.dirname(__file__)
_BENCHMARK_PATH = f"{_dir}/benchmark_apps"
_BENCHMARK_STATS_PATH = f"{_dir}/benchmark_stats"
_BENCHMARK_LOG_PATH = f"{_dir}/benchmark_logs"

_benchmark_paths = {
    "Renaissance": f"{_BENCHMARK_PATH}/renaissance-gpl-0.15.0.jar",
}


def get_benchmark_log_path(gc: str, benchmark_group: str, benchmark_name: str) -> str:
    return f"{_BENCHMARK_LOG_PATH}/{benchmark_group}_{benchmark_name}_{gc}.log"


def get_benchmark_stats_path(gc: str, benchmark_group: str, benchmark_name: str) -> str:
    return f"{_BENCHMARK_STATS_PATH}/{benchmark_group}_{benchmark_name}_{gc}.json"


def get_gc_global_stats_path(gc: str) -> str:
    return f"{_BENCHMARK_STATS_PATH}/global_stats/{gc}.json"


def get_benchmark_jar_path(benchmark_group: str) -> str:
    return _benchmark_paths[benchmark_group]


def is_cpu_intensive(cpu: float) -> bool:
    return cpu >= _CPU_THRESHOLD


def load_benchmark_results():
    return [
        BenchmarkResult.load_from_json(i)
        for i in glob.glob(f"{_BENCHMARK_STATS_PATH}/*.json")
    ]
