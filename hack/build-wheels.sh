#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

ROOT_FOLDER="$(dirname "${0}")/.."

if [ "$#" -lt 1 ]; then
  echo 'Invalid number of arguments'
  echo "Usage: ./build-wheels.sh <outputPath> [<runtimes>]"
  exit 1
fi

_buildWheel() {
  local _srcPath=$1
  local _outputPath=$2

  if [ -n "$(find "$_srcPath" -maxdepth 1 -type l 2>/dev/null)" ]; then
    # Symlinks present (e.g. onnx-cuda): poetry-core cannot build with
    # symlinks outside the project root. Build from a dereferenced temp
    # copy to avoid mutating the source tree.
    local _tmpDir
    _tmpDir=$(mktemp -d)
    cp -rL "$_srcPath/." "$_tmpDir/"
    rm -rf "$_tmpDir/.tox" "$_tmpDir/.venv" "$_tmpDir/dist" \
           "$_tmpDir/.envs" "$_tmpDir/.metrics"
    pushd "$_tmpDir"
    poetry build
    cp "$_tmpDir/dist/"* "$_outputPath"
    popd
    rm -rf "$_tmpDir"
  else
    # No symlinks: build in place (original behaviour)
    # Poetry doesn't let us send the output to a separate folder so we'll
    # `cd` into the folder and then move the wheels out
    # https://github.com/python-poetry/poetry/issues/3586
    pushd "$_srcPath"
    poetry build
    local _currentDistPath=$PWD/dist
    if ! [[ "$_currentDistPath" = "$_outputPath" ]]; then
      cp $_currentDistPath/* "$_outputPath"
    fi
    popd
  fi
}

_main() {
  # Convert any path into an absolute path
  local _outputPath=$1
  local _runtimes="${2:-}"
  mkdir -p $_outputPath
  if ! [[ "$_outputPath" = /* ]]; then
    pushd $_outputPath
    _outputPath="$PWD"
    popd
  fi

  # Build MLServer
  echo "---> Building MLServer wheel"
  _buildWheel . $_outputPath

  if [ -n "$_runtimes" ]; then
    for _runtime_name in $_runtimes; do
      local _runtime_path="$ROOT_FOLDER/runtimes/$_runtime_name"
      if [ -d "$_runtime_path" ]; then
        echo "---> Building MLServer runtime: '$_runtime_name'"
        _buildWheel "$_runtime_path" "$_outputPath"
      else
        echo "Warning: runtime '$_runtime_name' does not exist in $ROOT_FOLDER/runtimes/"
      fi
    done
  else
    for _runtime in "$ROOT_FOLDER/runtimes/"*; do
      echo "---> Building MLServer runtime: '$_runtime'"
      _buildWheel $_runtime $_outputPath
    done
  fi
}

_main "$1" "${2:-}"
