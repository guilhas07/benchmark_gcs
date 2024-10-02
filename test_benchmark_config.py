import unittest
from unittest.mock import mock_open, patch

from benchmark_config import BenchmarkSuiteCollection


class TestBenchmarkConfig(unittest.TestCase):
    def test_invalid_benchmark_suites(self):
        tests = {
            "Invalid JSON": (
                "",
                "Couldn't read file_path='test.json': Expecting value: line 1 column 1 (char 0).",
            ),
            "Non Existing": (
                "{}",
                "Error: Please provide at least one benchmark suite to test.",
            ),
            "String Value": (
                '{"benchmark_suites": "teste"}',
                'Error: benchmark_suites should be a list instead of "<class \'str\'>" with value "teste".',
            ),
            "Dict Value": (
                '{"benchmark_suites": {}}',
                'Error: benchmark_suites should be a list instead of "<class \'dict\'>" with value "{}".',
            ),
        }

        for name, (data, msg_expected) in tests.items():
            with patch("builtins.open", mock_open(read_data=data)):
                with self.assertRaises(
                    AssertionError,
                    msg=f'Failed: Test {name} should\'ve raised an assertion error with msg="{msg_expected}"',
                ) as cm:
                    BenchmarkSuiteCollection.load_from_json("test.json")

                self.assertEqual(
                    str(cm.exception),
                    msg_expected,
                    f'Failed: Test {name} got message "{cm.exception}" instead of "{msg_expected}"',
                )

    # def test_config_handles_white_spaces(self):
    #    data = """
    #     PROJECT_IDS = "project1"
    #       EMAIL = "test@gmail.com"
    #      PASSWORD =  "test"
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        config = Config()
    #        self.assertEqual(config.project_ids, {"project1"})
    #        self.assertEqual(config.email, "test@gmail.com")
    #        self.assertEqual(config.password, "test")

    # def test_config_handles_comments(self):
    #    data = """
    #    PROJECT_IDS="project1"#comment
    #    EMAIL="test@gmail.com"  # comment
    #    PASSWORD="test" # comment
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        config = Config()
    #        self.assertEqual(config.project_ids, {"project1"})
    #        self.assertEqual(config.email, "test@gmail.com")
    #        self.assertEqual(config.password, "test")

    # def test_config_handles_white_spaces_and_comments(self):
    #    data = """
    #     PROJECT_IDS = "project1"#comment
    #       EMAIL = "test@gmail.com"   #comment
    #      PASSWORD =  "test"   #comment
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        config = Config()
    #        self.assertEqual(config.project_ids, {"project1"})
    #        self.assertEqual(config.email, "test@gmail.com")
    #        self.assertEqual(config.password, "test")

    # def test_config_handles_multiple_projects(self):
    #    data = """
    #    PROJECT_IDS="project1,project2"
    #    EMAIL="test@gmail.com"
    #    PASSWORD="test"
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        config = Config()
    #        self.assertEqual(config.project_ids, {"project1", "project2"})
    #        self.assertEqual(config.email, "test@gmail.com")
    #        self.assertEqual(config.password, "test")

    # def test_config_ignore_invalid_keys(self):
    #    data = """
    #    PROJECT_IDS="project1"
    #    EMAIL="test@gmail.com"
    #    PASSWORD="test"
    #    INVALID="invalid"
    #    INVALID2=12345"testTest"12345
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        config = Config()
    #        self.assertEqual(config.project_ids, {"project1"})
    #        self.assertEqual(config.email, "test@gmail.com")
    #        self.assertEqual(config.password, "test")

    # def test_config_handles_all_valid_characters(self):
    #    data = """
    #    PROJECT_IDS="az0123456789"
    #    EMAIL="tEsT@gmail.com"
    #    PASSWORD="azAZ0123456789@#$%^&+=* "
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        config = Config()
    #        self.assertEqual(config.project_ids, {"az0123456789"})
    #        self.assertEqual(config.email, "tEsT@gmail.com")
    #        self.assertEqual(config.password, "azAZ0123456789@#$%^&+=* ")

    # def test_config_fails_if_email_invalid(self):
    #    data = """
    #    PROJECT_IDS="test"
    #    PASSWORD="test"
    #    EMAIL=
    #    EMAIL=""
    #    EMAIL="???@gmail.com"
    #    EMAIL=" @gmail.com"
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        self.assertRaises(AssertionError, Config)

    # def test_config_fails_if_password_invalid(self):
    #    data = """
    #    PROJECT_IDS="test"
    #    EMAIL="test"
    #    PASSWORD=""
    #    PASSWORD=
    #    PASSWORD="???"
    #    """
    #    with patch("builtins.open", mock_open(read_data=data)):
    #        self.assertRaises(AssertionError, Config)


if __name__ == "__main__":
    # with patch("builtins.print"):
    unittest.main()
