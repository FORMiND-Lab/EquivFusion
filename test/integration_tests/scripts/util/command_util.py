import subprocess
from pathlib import Path

class CommandUtil:
    @staticmethod
    def execute_shell_script(
            shell_env,
            case_dir: str,
            shell_path: str,
            output_path: str,
            timeout
    ):
        """Execute shell script"""
        # Construct command
        cmd = ["bash", shell_path]
        # Start subprocess
        process = subprocess.Popen(
            cmd,
            env=shell_env,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True  # Use text mode instead of binary
        )

        try:
            # Wait for completion with timeout
            stdout, _ = process.communicate(timeout=timeout)

            # Write output to file
            output_file = Path(output_path)
            output_file.write_text(stdout)

            return "success" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            # Kill subprocess
            process.kill()
            return "timeout"
        except Exception as e:
            return "failed"


