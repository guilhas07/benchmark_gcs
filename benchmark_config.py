from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Optional
import string


@dataclass
class BenchmarkRunOptions:
    command: Optional[str] = None
    extra_java_options: Optional[str] = None
    post_exec_script: Optional[str] = None

    def __post_init__(self):
        # NOTE: This class allows values with type <class 'str'> and <class 'NoneType'>
        allowed_types = [type(""), type(None)]
        for field_name in self.__dataclass_fields__:
            v = getattr(self, field_name)
            assert (
                type(v) in allowed_types
            ), f"Error: {field_name} with type {type(v)} should have type in {allowed_types}."


@dataclass
class BenchmarkConfig:
    name: str
    run_options: BenchmarkRunOptions | None

    def __init__(self, name: str, run_options: dict):
        self.name = name
        self.run_options = BenchmarkRunOptions(**run_options)

    def __post_init__(self):
        validate_str(getattr(self, "name"), self.__class__.__name__, "name")


@dataclass
class BenchmarkSuite:
    """
    Represents a suite of benchmarks to be executed

    Attributes:
        suite_name: Name used to save log/stats files
        jar_path: Path to your suite jar
        run_options: Options to run your benchmarks
        benchmarks_config: A list of individual benchmark configurations.
            If this is specified your suite will use the benchmark name like this:
            (default java options) (extra_java_options) ./jar_path **benchmark_name** (benchmark_options)
            NOTE: It is possible to override every benchmark option in BenchmarkSuite by specifying it in `BenchmarkConfig`
    """

    suite_name: str
    jar_path: str
    run_options: BenchmarkRunOptions | None
    benchmarks_config: list[BenchmarkConfig] | None

    @staticmethod
    def from_dict(data: dict) -> BenchmarkSuite:
        assert isinstance(
            data, dict
        ), f"Error: data should be a dictionary instead of {type(data)}."

        suite_name = data.get("suite_name")
        validate_str(suite_name, BenchmarkSuite.__name__, "suite_name")

        allowed_chars = string.ascii_letters + string.digits + " -_"
        for c in suite_name:  # pyright: ignore
            if c not in allowed_chars:
                raise AssertionError(
                    f"Error: '{suite_name}' has invalid charactes. Allowed Characters: {allowed_chars}."
                )

        jar_path = data.get("jar_path")
        validate_str(jar_path, BenchmarkSuite.__name__, "jar_path")
        run_options = data.get("run_options")
        assert isinstance(
            run_options, dict
        ), f"Error: run_options should be a dictionary instead of {type(run_options)}"
        run_options = BenchmarkRunOptions(**run_options)
        benchmark_config = data.get("benchmark_config")
        return BenchmarkSuite(suite_name, jar_path, run_options, benchmark_config)


@dataclass
class BenchmarkSuiteCollection:
    benchmark_suites: list[BenchmarkSuite]

    @staticmethod
    def load_from_json(file_path: str) -> BenchmarkSuiteCollection:
        with open(file_path) as f:
            try:
                data = json.loads(f.read())
            except Exception as e:
                raise AssertionError(f"Couldn't read {file_path=}: {e}.")

            suites = data.get("benchmark_suites", [])

            assert isinstance(
                suites, list
            ), f'Error: benchmark_suites should be a list instead of "{type(suites)}" with value "{suites}".'

            assert (
                len(suites) > 0
            ), "Error: Please provide at least one benchmark suite to test."

            return BenchmarkSuiteCollection(
                benchmark_suites=[
                    BenchmarkSuite.load_from_json(suite) for suite in suites
                ]
            )


def validate_str(value: Any, class_name: str, field_name: str):
    """Check if given value is a string with length > 0.
        Raises AssertionError with msg if it's not the case.

    Args:
        value: Value to validate
        field_name: Field being validated
    """
    err_str_msg = "Error: %s in %s should've a non null string-value, with one or more characters."
    assert value is not None and isinstance(value, str) and len(value) > 0, (
        err_str_msg % (field_name, class_name)
    )


# a = """ {
#    "DaCapo": {
#        "kafka": { "java": ["-XX:+ExitOnOutOfMemoryError"] },
#        "cassandra": {
#            "java": ["-Djava.security.manager=allow"]
#        },
#        "h2o": {
#            "java": ["-Dsys.ai.h2o.debug.allowJavaVersions=21"]
#        }
#    }
# }"""
#
# a = BenchmarkSuiteCollection.load_from_json(a)
