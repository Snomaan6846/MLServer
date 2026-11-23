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

  # Build into a fresh per-invocation directory: it contains only
  # this invocation's artifacts (no stale wheel from an earlier
  # build can be mistaken for a fresh one), and the shared output
  # folder is never wiped (wheels from earlier invocations survive).
  local _tmpDist
  _tmpDist="$(mktemp -d)"

  pushd $_srcPath > /dev/null
  poetry build --output "${_tmpDist}"
  popd > /dev/null

  # The wheels must be pure and self-sufficient for image-time
  # compilation: the C source rides the core wheel, a compiled
  # library never rides any wheel (each image build compiles it into
  # the installed tree — per-platform by construction). These
  # assertions fail loudly if the pyproject include config regresses.
  local _whl _listing _coreWheel
  for _whl in "${_tmpDist}"/*.whl; do
    _listing="$(python -m zipfile -l "${_whl}")"
    if grep -q '\.so' <<<"$_listing"; then
      echo "ERROR: ${_whl} contains a compiled library — wheels must stay pure" >&2
      exit 1
    fi
  done
  _coreWheel="$(ls "${_tmpDist}"/mlserver-*.whl 2> /dev/null || true)"
  if [ -n "${_coreWheel}" ]; then
    _listing="$(python -m zipfile -l "${_coreWheel}")"
    if ! grep -q 'fastspan/parser\.c' <<<"$_listing"; then
      echo "ERROR: ${_coreWheel} is missing fastspan/parser.c (the image-time compile source)" >&2
      exit 1
    fi
  fi

  mv "${_tmpDist}"/* "${_outputPath}"/
  rmdir "${_tmpDist}"
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
