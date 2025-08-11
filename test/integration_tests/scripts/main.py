import sys
from util.parser_util import ParserUtil
from test import run_test

if __name__ == "__main__":
    try:
        # 1. Parse args
        args = ParserUtil.parse_args()

        # 2. Run test
        failed_count = run_test(args)
        exit_code = 1 if failed_count != 0 else 0

    except Exception as e:
        print(f"Critical error: {str(e)}", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)
