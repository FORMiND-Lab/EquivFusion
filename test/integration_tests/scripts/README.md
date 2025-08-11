# Overview

# Directory Structure
```
test_scripts/
├── util/                       # Utility modules
│   ├── path_util.py            # Path handling utilities
│   ├── log_util.py             # Logging utilities
│   ├── cases_collect_util.py   # Test case collection utilities
│   ├── compare_util.py         # Result comparison utilities
│   ├── command_util.py         # Command execution utilities
│   └── parser_util.py          # Command-line argument parsing
├── test.py                     # Core test execution logic
├── main.py                     # Entry point (parses arguments and starts tests)
└── README.md
```

# Test Case Requirements
```
cases_dir/
├── case1/
│   ├── case1.sh
│   └── case1.golden
└── ...
```

# Usage
## Basic command
```bash
python3 main.py --cases-dir <cases-dir> --tool-dir <tool-dir> [--log-dir <log_dir> --theads <threads> --cases <cases-list or cases-file> --timeout <timeout>] 
```

## Examples
### 1. Run all test cases
```bash
python3 main.py --cases-dir test_cases --tool-dir build/bin
```
### 2. Run specific cases (case1 and case2):
```bash
python3 main.py --cases-dir test_cases --tool-dir build/bin --cases case1 case2
```
### 3. Run cases listed in a file
```bash
python3 main.py --cases-dir test_cases --tool-dir build/bin --cases cases.txt
```

## Command-Line Arguments
| Argument      | Required | Description                                                                                                          |
|---------------| -------- |----------------------------------------------------------------------------------------------------------------------|
| `--cases-dir` | Yes      | Cases directory                                                                                                      |
| `--tool-dir`  | Yes      | Tool directory (added to `PATH` for script access)                                                                   |
| `--log-dir`   | No       | Log directory (default: `./log`)                                                                                     |
| `--cases`     | No       | Filter test cases: - Space-separated case names (e.g., `case1 case2`) - Path to a list file (one case name per line) |
| `--threads`   | No       | Number of threads for parallel execution (default: 4)                                                                |
| `--timeout`   | No       | Timeout for each case (default: 300)                                                                                 |


# Log
- Log files are stored in the directory specified by --log-dir, including：
  - summary log: {log-dir}/result.log
  - single-case logs: {log-dir}/{case-name}/{case-name}.log


