import os
import argparse

class ParserUtil:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Run integration tests and compare results with golden files.")

        parser.add_argument(
            "--cases-dir", required=True,
            help="Cases directory."
        )
        parser.add_argument(
            "--tool-dir", required=True,
            help="Tool Directory."
        )
        parser.add_argument(
            "--log-dir", default="log",
            help="Log directory. Default is 'log'."
        )
        parser.add_argument(
            "--cases", nargs="*",  # 接受0个或多个参数
            help="Test cases to run: either space-separated case names (e.g. case1 case2) or a path to a case list file"
        )
        parser.add_argument(
            "--threads", type=int, default=4,
            help="Number of threads to use for parallel execution (default: 4)"
        )
        parser.add_argument(
            "--timeout", type=int, default=300,
            help="Timeout for each case (default: 300)"
        )

        return  parser.parse_args()
