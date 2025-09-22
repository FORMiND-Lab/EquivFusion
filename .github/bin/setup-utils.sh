#!/usr/bin/bash

DEPS_DIR="$(pwd)/deps"

mkdir -p "${DEPS_DIR}"

function download_github
{
  local repo="$1"
  local version="$2"
  local location="$3"
  local name=$(echo "$repo" | cut -d '/' -f 2)
  local archive="$name-$version.tar.gz"

  curl -o "$archive" -L "https://github.com/$repo/archive/$version.tar.gz"

  rm -rf "${location}"
  tar xfz "$archive"
  rm "$archive"
  mv "$name-$version" "${location}"
}