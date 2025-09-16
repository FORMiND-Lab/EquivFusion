#!/usr/bin/bash

set -e -o pipefail

source "$(dirname "$0")/setup-utils.sh"

REPO="arminbiere/aiger"
DIR=${DEPS_DIR}/aiger
COMMIT_ID="57594d2f95b286289da02ea37e2c3c934893dff5"

download_github "$REPO" "$COMMIT_ID" "$DIR"
cd "$DIR"

# Build
./configure.sh && make
