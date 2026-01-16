#!/bin/bash

declare -A CONFIG_z3=(
    [repo]="Z3Prover/z3"
    [commit_id]="9232ef579cb3df28368a8dd95613dc8d06260d42"  # release z3-4.15.0
)

build_z3() {
    mkdir build && cd build
    cmake ..; make
}

install_z3() {
    cp ${SOLVERS_DIR}/z3/build/z3 ${INSTALL_DIR} || return 1
}
