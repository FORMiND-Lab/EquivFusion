#!/usr/bin/bash

set -e -o pipefail

source "$(dirname "$0")/setup-utils.sh"

REPO="Boolector/boolector"
DIR=${DEPS_DIR}/boolector
COMMIT_ID="393cdfba3735d334bb4e6525500b8a0280dd41e6"

download_github "$REPO" "$COMMIT_ID" "$DIR"
cd "$DIR"

# Download and build Lingeling
./contrib/setup-lingeling.sh

# Download and build BTOR2Tools
./contrib/setup-btor2tools.sh

# Build Boolector
./configure.sh && cd build && make


