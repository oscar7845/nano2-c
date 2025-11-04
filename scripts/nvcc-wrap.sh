#!/usr/bin/env bash
# Forward everything to nvcc but force GCC 14 as the host C++ compiler.
exec /usr/local/cuda-12.9/bin/nvcc -ccbin /usr/bin/g++-14 "$@"
