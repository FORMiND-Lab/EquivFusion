# Overview

# Directory Structure
```
scripts/
├── util/                       # Utility modules
│   ├── cases_collect_util.py   # Test case collection utilities
│   ├── command_util.py         # Command execution utilities
│   ├── log_util.py             # Logging utilities
│   ├── parser_util.py          # Command-line argument parsing
│   └── path_util.py            # Path handling utilities
├── main.py                     # Entry point (parses arguments and starts tests)
├── test.py                     # Core test execution logic
├── test_run_gui.py             # GUI
└── README.md
```

# Test Case Requirements
```
cases/
├── example_test_001/
│   └── example_test_001.sh     # shell script include CHECK
└── ...
```

# Usage
## Basic command
```bash
python3 main.py [--tool-dir <tool-dir>] [--llvm-dir <llvm-dir>] [--cases-dir <cases-dir>] [--log-dir <log-dir>] [--cases <cases-list or cases-file] [--threads <threads>] [--timeout <timeout>]
```

## Examples
### 1. Run all test cases
```bash
python3 main.py --cases-dir test_cases
```
### 2. Run specific cases (case1 and case2):
```bash
python3 main.py --cases-dir test_cases --cases case1 case2
```
### 3. Run cases listed in a file
```bash
python3 main.py --cases-dir test_cases --cases cases.txt
```

## Command-Line Arguments
| Argument      | Required | Description                                                                                                                    |
|---------------|----------|--------------------------------------------------------------------------------------------------------------------------------|
| `--tool-dir`  | No       | TOOL directory (added to `PATH` for script access)      (default: project_directory/build/bin)                                 |
| `--llvm-dir`  | No       | LLVM directory (added to `PATH` for script access)      (default: project_directory/circt_prebuild/bin)                        |
| `--cases-dir` | No       | Cases directory                                         (default: project_directory/test/integration_tests/cases)              |
| `--log-dir`   | No       | Log directory                                           (default: project_directory)                                           |
| `--cases`     | No       | Filter test cases: <br/>- Space-separated case names (e.g., `case1 case2`) <br/>- Path to a list file (one case name per line) |
| `--threads`   | No       | Number of threads for parallel execution                (default: 4)                                                           |
| `--timeout`   | No       | Timeout for each case                                   (default: 300)                                                         |


# Log
- Log files are stored in the directory specified by --log-dir, including：
  - summary log: {log-dir}/log/result.log
  - single-case logs:
    - execute result: {log-dir}/log/{case-name}/{case-name}.log
    - filecheck result: {log-dir}/log/{case-name}/{case-name}_filecheck.log


