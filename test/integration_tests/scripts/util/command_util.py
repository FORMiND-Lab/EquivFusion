import subprocess
from pathlib import Path

class CommandUtil:
    @staticmethod
    def execute_shell_script(
            shell_env,
            case_dir: str,
            shell_path: Path,
            output_path: Path,
            timeout
    ):
        """Execute shell script"""
        # Construct command
        cmd = ["bash", str(shell_path)]
        # Start subprocess
        process = subprocess.Popen(
            cmd,
            env=shell_env,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True  # Use text mode instead of binary
        )

        # create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Wait for completion with timeout
            stdout, _ = process.communicate(timeout=timeout)

            # Write output to file
            output_path.write_text(stdout)

            return "success" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            # Kill subprocess
            process.kill()

            # Read partial stdout from the process pipe
            error_msg = (f"Command: {' '.join(cmd)}\n"
                         f"Timeout: Command timed out after {timeout} seconds\n")
            output_path.write_text(error_msg)
            return "timeout"
        except Exception as e:
            error_msg = (f"Command: {' '.join(cmd)}\n"
                         f"Exception Error: {str(e)}")
            output_path.write_text(error_msg)
            return "failed"

    @staticmethod
    def execute_file_check(shell_env,
                           output_path: Path,
                           check_path: Path,
                           check_output: Path):
        cmd = ["FileCheck", "--input-file", str(output_path), str(check_path)]
        process = subprocess.Popen(
            cmd,
            env=shell_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True  # Use text mode instead of binary
        )

        try:
            stdout, _ = process.communicate()

            if process.returncode == 0:
                return "success"
            else:
                # Write error output to check_output
                check_output.write_text(stdout)
                return "failed"
        except Exception as e:
            error_msg = (f"Command: {' '.join(cmd)}\n"
                         f"Exception Error: {str(e)}")
            check_output.write_text(error_msg)
            return "failed"
