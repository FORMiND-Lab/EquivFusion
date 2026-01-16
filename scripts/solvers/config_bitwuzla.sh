#!/bin/bash

declare -A CONFIG_bitwuzla=(
    [repo]="bitwuzla/bitwuzla"
    [commit_id]="61bdc637a291c27597fc04263328ecd29d36cc12" # Release 0.8.2
)

build_bitwuzla() {
    # Configure Bitwuzla release build
    ./configure.py || return 1

    # Build
    cd build && ninja
}

install_bitwuzla() {
    cp ${SOLVERS_DIR}/bitwuzla/build/src/main/bitwuzla ${INSTALL_DIR}
}

