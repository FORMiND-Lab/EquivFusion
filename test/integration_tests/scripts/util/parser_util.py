import os
import argparse

DEFAULT_TOOL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "build/bin"))
DEFAULT_LLVM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../", "circt_prebuild/bin"))

DEFAULT_CASES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "cases"))
DEFAULT_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
DEFAULT_THREADS = 4
DEFAULT_TIMEOUT = 300

class ParserUtil:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Run integration tests and compare results with golden files.")

        parser.add_argument("--tool-dir",   default = DEFAULT_TOOL_DIR,     help = f"Tool directory  (default : '{DEFAULT_TOOL_DIR}'")
        parser.add_argument("--llvm-dir",   default = DEFAULT_LLVM_DIR,     help = f"LLVM directory. (default : '{DEFAULT_LLVM_DIR}'")
        parser.add_argument("--cases-dir",  default = DEFAULT_CASES_DIR,    help = f"Cases directory (default : '{DEFAULT_CASES_DIR}'")
        parser.add_argument("--log-dir",    default = DEFAULT_LOG_DIR,      help = f"Log directory   (default: '{DEFAULT_LOG_DIR}'")

        parser.add_argument(
            "--cases",
            nargs = "*",  # 接受0个或多个参数
            help = "Test cases to run: either space-separated case names (e.g. case1 case2) or a path to a case list file"
        )
        parser.add_argument(
            "--threads",
            type = int,
            default = DEFAULT_THREADS,
            help = f"Number of threads to use for parallel execution (default: '{DEFAULT_THREADS}')"
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default = DEFAULT_TIMEOUT,
            help = f"Timeout for each case (default: '{DEFAULT_TIMEOUT}')"
        )

        return  parser.parse_args()
