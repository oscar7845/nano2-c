#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Usage examples:
#   NP=2 ./scripts/run_mpi.sh
#   ./scripts/run_mpi.sh -H node1,node2 -N 2

NP="${NP:-2}"
HOSTS_FLAG=""
EXTRA=()

if [[ "${1:-}" == "-H" ]]; then
  HOSTS_FLAG="--host $2"
  shift 2
fi
if [[ "${1:-}" == "-N" ]]; then
  NP="$2"
  shift 2
fi
if [[ $# -gt 0 ]]; then
  EXTRA=("$@")
fi

# CUDA-aware UCX path (Open MPI 5 + UCX)
export OMPI_MCA_pml=ucx
# UCX hints for GPU staging; adjust to your fabric if needed
export UCX_TLS=${UCX_TLS:-"sm,cuda_copy,cuda_ipc,rc"}
export UCX_GPU_COPY_MODE=${UCX_GPU_COPY_MODE:-"cuda"}
# Some sites need to disable old BTLs explicitly
export OMPI_MCA_btl=${OMPI_MCA_btl:-"self,vader"}

mpirun -np "${NP}" ${HOSTS_FLAG} ./build/nano2 "${EXTRA[@]}"

