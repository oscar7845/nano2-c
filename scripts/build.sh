#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# H100 default; override with: CMAKE_CUDA_ARCHITECTURES=86 ./scripts/build.sh
: "${CMAKE_CUDA_ARCHITECTURES:=90}"

mkdir -p build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DNANO2_ALLOW_UNSUPPORTED_GCC=ON
cmake --build build -j

