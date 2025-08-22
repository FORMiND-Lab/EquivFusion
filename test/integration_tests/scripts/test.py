import os
import sys
import concurrent.futures
from pathlib import Path
from util.path_util import PathUtil
from util.log_util import LogUtil
from util.cases_collect_util import CasesCollectUtil
from util.command_util import CommandUtil

LOG_DIRECTORY_NAME = "log"

class TestRunner:
    """Test executor class encapsulating test case execution logic"""

    def __init__(self, args):
        """Initialize test configuration"""
        self.tool_dir = PathUtil.to_absoluate(args.tool_dir)
        self.llvm_dir = PathUtil.to_absoluate(args.llvm_dir)

        self.cases_dir = PathUtil.to_absoluate(args.cases_dir)
        self.log_dir = PathUtil.to_absoluate(args.log_dir)
        self.cases_spec = args.cases
        self.threads = args.threads
        self.timeout = args.timeout

        # Setup environment variables
        self._setup_environment_variables()

        actual_log_dir = Path(PathUtil.to_absoluate(args.log_dir)) / LOG_DIRECTORY_NAME
        PathUtil.ensure_dir_exists(str(actual_log_dir))
        # Init log
        self.logger = LogUtil.init_logger(actual_log_dir )

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
        self.shell_env["PATH"] = f"{self.llvm_dir}:{self.shell_env.get('PATH', '')}"
        self.shell_env["PATH"] = f"{self.tool_dir}:{self.shell_env.get('PATH', '')}"


    def handle_single_case_result(self, log_prefix, result, output_path, check_path, check_output):
        def handle_success() -> bool:
            check_result = CommandUtil.execute_file_check(self.shell_env, output_path, check_path, check_output)
            if check_result == "success":
                self.logger.info(f"{log_prefix} Passed")
                return True
            else:
                self.logger.info(f"{log_prefix} Failed [FileCheck error, see {check_output}]")
                return False

        def handle_timeout() -> bool:
            self.logger.info(f"{log_prefix} Failed [timeout]")
            return False

        def handle_failed() -> bool:
            self.logger.info(f"{log_prefix} Failed [run script error, see {output_path}]")
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
        log_prefix = f"    {index}/{total_count} - {case_name} :"

        try:
            case_test_path = Path(case_dir) / f"{case_name}.sh"
            output_path = Path(self.log_dir) / LOG_DIRECTORY_NAME / case_name / f"{case_name}.log"

            if not os.path.exists(case_test_path):
                self.logger.info(f"{log_prefix} Failed [no case test file]")
                return False

            # Execute test case
            exec_result = CommandUtil.execute_shell_script(
                self.shell_env,
                case_dir,
                case_test_path,
                output_path,
                self.timeout
            )

            check_output = Path(self.log_dir) / LOG_DIRECTORY_NAME / case_name  / f"{case_name}_filecheck.log"
            # Analysis test result
            return self.handle_single_case_result(log_prefix, exec_result, output_path, case_test_path, check_output)

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
        self.logger.info(f"    TOOL directory   : {self.tool_dir}")
        self.logger.info(f"    LLVM directory   : {self.llvm_dir}")
        self.logger.info(f"    log directory    : {self.log_dir}")
        self.logger.info(f"    cases directory  : {self.cases_dir}")
        self.logger.info(f"    cases            : {self.cases_spec}")
        self.logger.info(f"    threads num      : {self.threads}")
        self.logger.info(f"    timeout          : {self.timeout}")
        self.logger.info("=" * 120 + "\n")

    def print_result(self, total_count, failed_count):
        """Print test summary results"""
        passed_count = total_count - failed_count
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        self.logger.info("\n" + "=" * 120)
        self.logger.info("Test Results Summary")
        self.logger.info("-" * 120)
        self.logger.info(f"    Total cases      : {total_count}")
        self.logger.info(f"    Passed           : {passed_count}")
        self.logger.info(f"    Failed           : {failed_count}")
        self.logger.info(f"    Pass rate        : {pass_rate:.2f}%")
        self.logger.info("=" * 120)

    def validate_used_paths(self) -> bool:
        """Validate used paths"""
        return all([
            PathUtil.validate_path_is_dir(self.logger, self.tool_dir, "TOOL directory"),
            PathUtil.validate_path_is_dir(self.logger, self.llvm_dir, "LLVM directory"),
            PathUtil.validate_path_is_dir(self.logger, self.log_dir, "Log directory"),
            PathUtil.validate_path_is_dir(self.logger, self.cases_dir, "Cases directory")
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