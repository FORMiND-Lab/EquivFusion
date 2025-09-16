#!/usr/bin/bash

set -e -o pipefail

source "$(dirname "$0")/setup-utils.sh"

REPO="arminbiere/kissat"
DIR=${DEPS_DIR}/kissat
COMMIT_ID="77bc7ea68afe80751a67df8561357f193e160fb1"

download_github "$REPO" "$COMMIT_ID" "$DIR"
cd "$DIR"

# Build
./configure && make

