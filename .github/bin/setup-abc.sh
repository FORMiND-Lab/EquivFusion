#!/usr/bin/bash

set -e -o pipefail

source "$(dirname "$0")/setup-utils.sh"

REPO="berkeley-abc/abc"
DIR=${DEPS_DIR}/abc
COMMIT_ID="9478c172881f7c392493420f639a940fdfeb9a00"

download_github "$REPO" "$COMMIT_ID" "$DIR"
cd "$DIR"

# Build
make
