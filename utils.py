import glob
import os
import subprocess
from typing import Optional

from model import BenchmarkResult

_CPU_THRESHOLD = 60

_dir = os.path.dirname(__file__)
_BENCHMARK_PATH = f"{_dir}/benchmark_apps"
_BENCHMARK_STATS_PATH = f"{_dir}/benchmark_stats"
_BENCHMARK_LOG_PATH = f"{_dir}/benchmark_logs"
_BENCHMARK_MATRIX_PATH = f"{_dir}/benchmark_matrix"

_benchmark_paths = {
    "Renaissance": f"{_BENCHMARK_PATH}/renaissance-gpl-0.15.0.jar",
}

_GRAAL = "Graal"
_HOTSPOT = "HotSpot"

_SUPPORTED_JDKS = [_GRAAL, _HOTSPOT]
_available_gcs = {
    _GRAAL: ["G1", "Parallel", "Z"],
    _HOTSPOT: ["G1", "Parallel", "Z"],
}


def get_cpu_count():
    cpu = os.cpu_count()
    assert cpu is not None, "Error: cpu_count is None"
    return cpu


def clean_logs_and_stats():
    print("Cleaning logs and stats directories")
    for i in glob.glob(f"{_BENCHMARK_STATS_PATH}/**/*.json", recursive=True):
        os.remove(i)
    for i in glob.glob(f"{_BENCHMARK_LOG_PATH}/**/*.log", recursive=True):
        os.remove(i)


def get_benchmark_log_path(
    gc: str, benchmark_group: str, benchmark_name: str, heap_size: str
) -> str:
    return f"{_BENCHMARK_LOG_PATH}/{benchmark_group}_{benchmark_name}_{gc}_{heap_size}m.log"


def get_benchmark_stats_path(
    gc: str,
    benchmark_group: str,
    benchmark_name: str,
    heap_size: str,
    jdk: str,
    error: Optional[str],
) -> str:
    return f"{_BENCHMARK_STATS_PATH}/{benchmark_group}_{benchmark_name}_{gc}_{heap_size}m_{jdk}_{'' if error is None else 'error'}.json"


def get_error_report_path(jdk: str, date: str) -> str:
    return f"{_BENCHMARK_STATS_PATH}/gc_stats/error_report_{jdk}_{date}.json"


def get_gc_stats_path(gc: str, jdk: str) -> str:
    return f"{_BENCHMARK_STATS_PATH}/gc_stats/{gc}_{jdk}.json"


def get_matrix_path(jdk: str, date: str) -> str:
    assert jdk in _SUPPORTED_JDKS, f"Runtime {jdk} not supported"
    return f"{_BENCHMARK_MATRIX_PATH}/{jdk}_{date}.json"


def get_benchmark_jar_path(benchmark_group: str) -> str:
    return _benchmark_paths[benchmark_group]


def is_cpu_intensive(cpu: float) -> bool:
    return cpu >= _CPU_THRESHOLD


def get_available_gcs() -> tuple[str, list[str]] | tuple[None, None]:
    """
    Get list of available garbage collectors for your java version.
    Only Graal and OpenJDK are currently supported.

    Returns:
        list[str] | None: List of garbage collectors. None if java version isn't supported.
    """

    p = subprocess.run(["java", "--version"], capture_output=True, text=True)
    print(f"Your java version is:\n {p.stdout}")
    for jdk in _SUPPORTED_JDKS:
        if jdk in p.stdout:
            return jdk, _available_gcs[jdk]

    return None, None


def get_heap_sizes() -> list[str]:
    """Returns a list of heap sizes in megabytes"""
    return ["256", "512", "1024", "2048", "4096", "8192"]


def load_benchmark_results(
    garbage_collector: str, heap_size: str, jdk: str
) -> list[BenchmarkResult]:
    return [
        BenchmarkResult.load_from_json(i)
        for i in glob.glob(
            f"{_BENCHMARK_STATS_PATH}/*{garbage_collector}_{heap_size}m_{jdk}*.json"
        )
    ]
