import os
import sys
import concurrent.futures
from typing import Tuple, Optional
from util.path_util import PathUtil
from util.log_util import LogUtil
from util.cases_collect_util import CasesCollectUtil
from util.compare_util import CompareUtil
from util.command_util import CommandUtil

class TestRunner:
    """Test executor class encapsulating test case execution logic"""

    def __init__(self, args):
        """Initialize test configuration"""
        # Ensure log directory exists
        PathUtil.ensure_dir_exists(args.log_dir)
        # Init log
        self.logger = LogUtil.init_logger(args.log_dir)

        self.cases_dir = PathUtil.to_absoluate(args.cases_dir)
        self.tool_dir = PathUtil.to_absoluate(args.tool_dir)
        self.log_dir = PathUtil.to_absoluate(args.log_dir)
        self.cases_spec = args.cases
        self.threads = args.threads
        self.timeout = args.timeout

        # Setup environment variables
        self._setup_environment_variables()

        # Print configuration information
        self.print_configure()

    def cleanup(self):
        """Clean up log handlers"""
        LogUtil.clean_logger(self.logger)

    def _setup_environment_variables(self) -> None:
        """Setup environment variables with absolute paths for shell execution"""
        # Create a copy of the current environment variables to avoid modifying global state
        self.shell_env = os.environ.copy()

        # absolute path of tool_dir add to PATH
        self.shell_env["PATH"] = f"{self.tool_dir}:{self.shell_env.get('PATH', '')}"

    def handle_single_case_result(self, log_prefix, result, output_path, golden_path):
        def handle_success() -> bool:
            # Check for golden file
            if not os.path.exists(golden_path):
                self.logger.info(f"{log_prefix} Failed [no golden file]")
                return False

            # Compare results
            match, diff = CompareUtil.compare_files(output_path, golden_path)
            if match:
                self.logger.info(f"{log_prefix} Passed")
                return True
            else:
                self.logger.info(f"{log_prefix} Failed [golden mismatch: {output_path}, {golden_path}]")
                return False

        def handle_timeout() -> bool:
            self.logger.info(f"{log_prefix} Failed [timeout]")
            return False

        def handle_failed() -> bool:
            self.logger.info(f"{log_prefix} Failed [run script error, see log: {output_path}]")
            return False

        result_handlers = {
            "success": handle_success,
            "timeout": handle_timeout,
            "failed": handle_failed
        }
        handler = result_handlers.get(result)
        if not handler:
            self.logger.warning(f"{log_prefix} Unknown result status: {result}")
            return False

        return handler()

    def run_single_case(self, case_dir: str, index: int, total_count: int) -> bool:
        """Execute a single test case"""
        case_name = os.path.basename(case_dir)
        log_prefix = f"\t{index}/{total_count} - {case_name} :"

        # Initialize log directory and output file
        case_log_dir = os.path.join(self.log_dir, case_name)
        PathUtil.ensure_dir_exists(case_log_dir)
        output_path = os.path.join(case_log_dir, f"{case_name}.log")

        try:
            shell_path = os.path.join(case_dir, f"{case_name}.sh")

            if not os.path.exists(shell_path):
                self.logger.info(f"{log_prefix} Failed [no script file]")
                return False

            # Execute test case
            self.logger.debug(f"{log_prefix} Executing shell script: {shell_path}")
            exec_result = CommandUtil.execute_shell_script(
                self.shell_env,
                case_dir,
                shell_path,
                output_path,
                self.timeout
            )

            golden_path = os.path.join(case_dir, f"{case_name}.golden")
            # Analysis test result
            return self.handle_single_case_result(log_prefix, exec_result, output_path, golden_path)

        except Exception as e:
            self.logger.info(f"{log_prefix} Failed [exception: {str(e)}]")
            return False

    def print_configure(self):
        """Print configuration information"""

        # Print full command
        command_parts = [sys.executable] + sys.argv[0:]
        full_command = " ".join(command_parts)
        self.logger.info(f'{full_command}')

        # Print configuration
        self.logger.info("=" * 120)
        self.logger.info("Test Configuration")
        self.logger.info("-" * 120)
        self.logger.info(f"\tcases directory    : {self.cases_dir}")
        self.logger.info(f"\ttool directory     : {self.tool_dir}")
        self.logger.info(f"\tlog directory      : {self.log_dir}")
        self.logger.info(f"\tcases              : {self.cases_spec}")
        self.logger.info(f"\tthreads num        : {self.threads}")
        self.logger.info(f"\ttimeout            : {self.timeout}")
        self.logger.info("=" * 120 + "\n")

    def print_result(self, total_count, failed_count):
        """Print test summary results"""
        passed_count = total_count - failed_count
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        self.logger.info("\n" + "=" * 120)
        self.logger.info("Test Results Summary")
        self.logger.info("-" * 120)
        self.logger.info(f"\tTotal cases        : {total_count}")
        self.logger.info(f"\tPassed             : {passed_count}")
        self.logger.info(f"\tFailed             : {failed_count}")
        self.logger.info(f"\tPass rate          : {pass_rate:.2f}%")
        self.logger.info("=" * 120)

    def validate_used_paths(self) -> bool:
        """Validate used paths"""
        return all([
            PathUtil.validate_path_is_dir(self.logger, self.cases_dir, "Cases directory"),
            PathUtil.validate_path_is_dir(self.logger, self.tool_dir, "Tool directory"),
            PathUtil.validate_path_is_dir(self.logger, self.log_dir, "Log directory")
        ])

    def run_all_tests(self) -> int:
        """Run all test cases"""
        try:
            # Validate used path
            if not self.validate_used_paths():
                return 0

            # Collect test cases
            case_dirs = CasesCollectUtil.collect_test_cases(self.logger, self.cases_dir, self.cases_spec)
            total_count = len(case_dirs)

            if total_count == 0:
                self.logger.info(f"No cases found...\n")
                return 0

            self.logger.info(f"Starting test...\n")

            # Execute tests in parallel
            failed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(
                        self.run_single_case,
                        case_dir,
                        index + 1,
                        total_count
                    )
                    for index, case_dir in enumerate(case_dirs)
                ]

                # Process results
                for future in concurrent.futures.as_completed(futures):
                    if not future.result():
                        failed_count += 1

            # Output summary
            self.print_result(total_count, failed_count)

            return failed_count

        except Exception as e:
            self.logger.error(f"Exception in run_all_tests(): {str(e)}", exc_info=True)
            return -1


def run_test(args) -> int:
    """Test execution entry point"""
    runner = TestRunner(args)
    res = runner.run_all_tests()
    runner.cleanup()
    return res