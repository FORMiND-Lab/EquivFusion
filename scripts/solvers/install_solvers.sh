#!/bin/bash

CURRENT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "${CURRENT_DIR}/config_solvers.sh"

SOLVERS_DIR=${SOLVERS_DIR:-"."}
mkdir -p "$SOLVERS_DIR"

download_solver()
{
    local name="$1"

    local config_file="${CURRENT_DIR}/config_${name}.sh"
    source "$config_file"
    local config_var="CONFIG_${name}"

    local repo=$(eval "echo \${${config_var}[repo]}")
    local commit_id=$(eval "echo \${${config_var}[commit_id]}")

    local archive_name="$name-$commit_id"
    local archive="$archive_name.tar.gz"

    cd "$SOLVERS_DIR"

    # remove exists solver
    rm -rf "$name"

    # download solver
    if ! curl -o "$archive" -L "https://github.com/$repo/archive/$commit_id.tar.gz"; then
        return 1
    fi

    # extract solver
    tar xfz "$archive" && rm "$archive"
    # move solver
    mv "$archive_name" "$name"

    return 0
}

execute_solver_function() {
    local name=$1
    local action=$2   # build/install

    local function="${action}_${name}"
    if ! "$function"; then
        return 1
    fi

    return 0
}

setup_solver() {
    local name=$1

    # download solver
    echo ">>> Downloading $name ..."
    if ! download_solver "$name"; then
        echo ">>> Download $name failed"
        return 1
    fi
    echo ">>> Download $name completed successfully"

    cd "$SOLVERS_DIR/$name"

    # build solver
    echo ">>> Building $name ..."
    if ! execute_solver_function "$name" "build"; then
      echo ">>> Build $name failed"
      return 1
    fi
    echo ">>> Build $name completed successfully"

    # install_solver
    if [ ! -z ${INSTALL_DIR} ]; then
        echo ">>> Installing $name ..."
        if ! execute_solver_function "$name" "install"; then
          echo ">>> Install $name failed"
          return 1
        fi
        echo ">>> Install $name completed successfully"
    fi

    return 0
}

main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi 
   
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            --*)
                name=${1:2}
                if is_supported_solver "$name"; then
                    if ! setup_solver "$name"; then
                        exit 1;
                    fi
                else
                    echo "Unsupport solvers: $name"
                    show_help
                    exit 1
                fi
                shift
                ;;
            *)
                echo "Unsupport option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

main "$@"


