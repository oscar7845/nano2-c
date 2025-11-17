#!/usr/bin/env bash
set -euo pipefail

#limit Open MPI BTLs to self + shared-memory (single-node runs)
export OMPI_MCA_btl=self,vader
unset OMPI_MCA_btl_tcp_if_include

#cd to repo root (script directory's parent), robust to symlinks
cd "$(dirname "${BASH_SOURCE[0]}")/.."

#run; make this non-fatal
./build/nano2 --config ./configs/nano2.json

