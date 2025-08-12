#!/usr/bin/bash

SEARCH_PATHS=(
    "/usr/lib"
    "/usr/lib64"
    "/usr/local/lib"
    "/usr/local/lib64"
)

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

Z3_PATH=$(find ${SEARCH_PATHS[@]} -name "libz3.so" -print -quit 2>/dev/null)

if [ -z "$Z3_PATH" ]; then
    echo "Error: Z3 library not found"
    exit 1
fi

EquivFusionLEC  $SCRIPT_DIR/input.mlir -c1 foo1 --c2 foo2 --run --shared-libs="$Z3_PATH"
