#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

ROOT_FOLDER="$(dirname "${0}")/.."
IMAGE_NAME="seldonio/mlserver"

if [ "$#" -ne 1 ]; then
  echo 'Invalid number of arguments'
  echo "Usage: ./build-images.sh <version>"
  exit 1
fi

_buildImage() {
  local _runtimes=$1
  local _tag=$2

  DOCKER_BUILDKIT=1 docker build $ROOT_FOLDER \
    --build-arg RUNTIMES="$_runtimes" \
    -t "$IMAGE_NAME:$_tag"
}

_assertNativeParser() {
  local _tag=$1

  # The image must carry the native JSON span parser: the Dockerfile
  # compiles it into the installed package tree (the wheel ships only
  # the C source), and enabling rest_fast_parse without the library
  # is a startup error. Catches a broken compile step or a packaging
  # regression before the image ships.
  docker run --rm "$IMAGE_NAME:$_tag" python3 -c \
    "from mlserver.codecs import fastspan; assert fastspan._load() is not None" \
    || {
      echo "ERROR: $IMAGE_NAME:$_tag does not contain libfastspan.so" >&2
      echo "Check the Dockerfile runtime-stage compile step and the pyproject parser.c include" >&2
      exit 1
    }
  echo "---> Asserted native parser in $_tag"
}

_buildRuntimeImage() {
  local _runtimePath=$1
  local _version=$2
  local _runtimeName=$(basename $_runtimePath)

  echo "---> Building MLServer runtime image: $_runtimeName"
  _buildImage "mlserver-$_runtimeName" "$_version-$_runtimeName"
}

_main() {
  local _version=$1

  echo "---> Building core MLServer images"
  _buildImage "all" $_version
  _assertNativeParser $_version
  _buildImage "" $_version-slim
  _assertNativeParser $_version-slim

  for _runtimePath in "$ROOT_FOLDER/runtimes/"*; do
    _buildRuntimeImage $_runtimePath $_version
    _assertNativeParser $_version-$(basename $_runtimePath)
  done
}

_main $1
