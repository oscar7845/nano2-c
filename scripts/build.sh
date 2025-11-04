#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Default arch for your RTX 3050; override like:
#   CMAKE_CUDA_ARCHITECTURES="86;89;90" ./scripts/build.sh
: "${CMAKE_CUDA_ARCHITECTURES:=86}"

NVCC_ARGS=()
if [[ -x "$PWD/scripts/nvcc-wrap.sh" && -x /usr/bin/g++-14 ]]; then
  NVCC_ARGS+=(-DCMAKE_CUDA_COMPILER="$PWD/scripts/nvcc-wrap.sh")
  NVCC_ARGS+=(-DCMAKE_CXX_COMPILER=/usr/bin/g++-14)
else
  echo "error: nvcc wrapper or g++-14 not found."
  echo "       Ensure /usr/bin/g++-14 exists and scripts/nvcc-wrap.sh is executable."
  exit 1
fi

mkdir -p build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "${NVCC_ARGS[@]}"

cmake --build build -j

