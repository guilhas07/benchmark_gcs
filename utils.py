import glob
import os
import subprocess

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

_GRAAL = "Graal"
_OPENJDK = "OpenJDK"

_SUPPORTED_JDKS = [_GRAAL, _OPENJDK]
_available_gcs = {
    _GRAAL: ["G1", "Parallel", "Z"],
    _OPENJDK: ["G1", "Parallel", "Z", "Shenandoah"],
}


def clean_logs_and_stats():
    print("Cleaning logs and stats directories")
    for i in glob.glob(f"{_BENCHMARK_STATS_PATH}/**/*.json", recursive=True):
        os.remove(i)
    for i in glob.glob(f"{_BENCHMARK_LOG_PATH}/**/*.log", recursive=True):
        os.remove(i)


def get_benchmark_log_path(
    gc: str, benchmark_group: str, benchmark_name: str, heap_size: int
) -> str:
    return f"{_BENCHMARK_LOG_PATH}/{benchmark_group}_{benchmark_name}_{gc}_{heap_size}m.log"


def get_benchmark_stats_path(
    gc: str, benchmark_group: str, benchmark_name: str, heap_size: int, success: bool
) -> str:
    success = "" if success else "_error"
    return f"{_BENCHMARK_STATS_PATH}/{benchmark_group}_{benchmark_name}_{gc}_{heap_size}m{success}.json"


def get_gc_global_stats_path(gc: str) -> str:
    return f"{_BENCHMARK_STATS_PATH}/global_stats/{gc}.json"


def get_benchmark_jar_path(benchmark_group: str) -> str:
    return _benchmark_paths[benchmark_group]


def is_cpu_intensive(cpu: float) -> bool:
    return cpu >= _CPU_THRESHOLD


def get_available_gcs() -> list[str] | None:
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
            return _available_gcs[jdk]

    return None


def get_heap_sizes() -> list[str]:
    """Returns a list of heap sizes in megabytes"""
    return ["256", "512", "1024", "2048", "4096", "8192"]


def load_benchmark_results(
    garbage_collector: str, heap_size: int
) -> list[BenchmarkResult]:
    return [
        BenchmarkResult.load_from_json(i)
        for i in glob.glob(
            f"{_BENCHMARK_STATS_PATH}/*{garbage_collector}_{heap_size}*.json"
        )
    ]
