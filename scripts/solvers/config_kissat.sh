#!/bin/bash

declare -A CONFIG_kissat=(
    [repo]="arminbiere/kissat"
    [commit_id]="77bc7ea68afe80751a67df8561357f193e160fb1"  # Release 4.0.3
)

build_kissat() {
    ./configure && make
}

install_kissat() {
    cp ${SOLVERS_DIR}/kissat/build/kissat ${INSTALL_DIR} || return 1
}
