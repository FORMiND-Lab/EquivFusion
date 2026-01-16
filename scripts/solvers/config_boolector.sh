#!/bin/bash

declare -A CONFIG_boolector=(
    [repo]="Boolector/boolector"
    [commit_id]="393cdfba3735d334bb4e6525500b8a0280dd41e6"  # Boolector 3.2.4
)

build_boolector() {
    # Download and build Lingeling
    ./contrib/setup-lingeling.sh || return 1

    # Download and build BTOR2Tools
    ./contrib/setup-btor2tools.sh || return 1

    # Build Boolector
    ./configure.sh && cd build && make || return 1
}

install_boolector() {
    cp ${SOLVERS_DIR}/boolector/build/bin/btormc ${INSTALL_DIR} || return 1
}
