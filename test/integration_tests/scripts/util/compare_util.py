import difflib
from typing import Tuple

class CompareUtil:
    @staticmethod
    def compare_files(output_path: str, golden_path: str) -> Tuple[bool, str]:
        """Compare file"""
        try:
            with open(output_path, "r") as output_file, \
                    open(golden_path, "r") as golden_file:

                output_lines = output_file.readlines()
                golden_lines = golden_file.readlines()

                if output_lines == golden_lines:
                    return True, ""

                # Generate diff report
                diff = difflib.unified_diff(
                    golden_lines,
                    output_lines,
                    fromfile="Expected",
                    tofile="Actual",
                    lineterm=""
                )
                return False, "\n".join(diff)

        except Exception as e:
            return False, f"Exception _compare_results(): {str(e)}"