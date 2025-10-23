#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")

equivfusion-hls "$CASE_DIR/design.mlir" -top "test"

#CHECK-NOT: hw.module @increment
