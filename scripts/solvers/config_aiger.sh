#!/bin/bash

declare -A CONFIG_aiger=(
    [repo]="arminbiere/aiger"
    [commit_id]="57594d2f95b286289da02ea37e2c3c934893dff5"
)

build_aiger() {
    ./configure.sh && make
}

install_aiger() {
    cp ${SOLVERS_DIR}/aiger/aigmiter ${INSTALL_DIR} || return 1
    cp ${SOLVERS_DIR}/aiger/aigtocnf ${INSTALL_DIR} || return 1
}
