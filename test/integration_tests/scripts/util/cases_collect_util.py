import os
from typing import List

class CasesCollectUtil:
    @staticmethod
    def load_case_names_from_file(logger, file_path: str) -> List[str]:
        """Load case names from a file (one per line, skip comments and empty lines)"""
        try:
            with open(file_path, "r") as f:
                cases = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # 跳过空行和注释
                        cases.append(line)
                return cases
        except Exception as e:
            logger.error(f"Failed to load cases from file {file_path}: {str(e)}")
            return []

    @staticmethod
    def collect_test_cases(logger, cases_dir, cases_spec) -> List[str]:
        """Collect test cases, with filtering based on --cases option"""
        # Get all available test case directories
        all_case_dirs = []
        for item in os.listdir(cases_dir):
            item_path = os.path.join(cases_dir, item)
            if os.path.isdir(item_path) and not item.startswith("."):
                all_case_dirs.append(item_path)

        # --cases not provided (cases_spec is None) - execute all cases
        if cases_spec is None:
            return sorted(all_case_dirs)

        # --cases provided without arguments (empty list) - execute no cases
        if len(cases_spec) == 0:
            return []

        # parse --cases parameter
        target_names = []
        if len(cases_spec) == 1:
            # Single parameter: could be file path or single case name
            candidate = cases_spec[0]
            if os.path.isfile(candidate):
                # file path: load cases from file
                target_names = CasesCollectUtil.load_case_names_from_file(logger, candidate)
            else:
                # single case name
                target_names = [candidate]
        else:
            # Multiple parameters: treat as list of case names
            target_names = cases_spec

        # Filter existing case directories
        filtered_dirs = []
        missing_cases = []
        for name in target_names:
            matched_dirs = [d for d in all_case_dirs if os.path.basename(d) == name]
            if matched_dirs:
                filtered_dirs.extend(matched_dirs)
            else:
                missing_cases.append(name)

        # Handle missing cases
        if missing_cases:
            logger.warning(f"Some cases not found: {missing_cases}")

        # 6. Remove duplicates and sort
        unique_filtered_dirs = sorted(list(set(filtered_dirs)))

        return unique_filtered_dirs