#!/usr/bin/bash

set -e -o pipefail

source "$(dirname "$0")/setup-utils.sh"

REPO="bitwuzla/bitwuzla"
DIR=${DEPS_DIR}/bitwuzla
COMMIT_ID="61bdc637a291c27597fc04263328ecd29d36cc12"

download_github "$REPO" "$COMMIT_ID" "$DIR"
cd "$DIR"

# Configure Bitwuzla release build
./configure.py

# Build
cd build && ninja


