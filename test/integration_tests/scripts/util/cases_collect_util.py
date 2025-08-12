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
        # 1. 获取所有可用的测试用例目录
        all_case_dirs = []
        for item in os.listdir(cases_dir):
            item_path = os.path.join(cases_dir, item)
            if os.path.isdir(item_path) and not item.startswith("."):
                all_case_dirs.append(item_path)

        # 2. 未指定--cases时，返回所有用例
        if not cases_spec:
            return sorted(all_case_dirs)

        # 3. 解析--cases参数（区分是文件还是用例名称列表）
        target_names = []
        if len(cases_spec) == 1:
            # 单个参数：可能是文件路径或单个用例名称
            candidate = cases_spec[0]
            if os.path.isfile(candidate):
                # 是文件路径：从文件加载用例名称
                target_names = CasesCollectUtil.load_case_names_from_file(logger, candidate)
            else:
                # 是单个用例名称
                target_names = [candidate]
        else:
            # 多个参数：视为用例名称列表
            target_names = cases_spec

        # 4. 筛选出存在的用例目录
        filtered_dirs = []
        missing_cases = []
        for name in target_names:
            matched_dirs = [d for d in all_case_dirs if os.path.basename(d) == name]
            if matched_dirs:
                filtered_dirs.extend(matched_dirs)
            else:
                missing_cases.append(name)

        # 5. 处理缺失的用例
        if missing_cases:
            logger.warning(f"Some cases not found: {missing_cases}")

        # 6. 去重并排序
        unique_filtered_dirs = sorted(list(set(filtered_dirs)))

        return unique_filtered_dirs