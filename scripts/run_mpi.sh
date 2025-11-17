#!/usr/bin/env bash
set -euo pipefail

#cd to repo root (robust to symlinks)
cd "$(dirname "${BASH_SOURCE[0]}")/.."

#MPI on a single node
export OMPI_MCA_btl=self,vader #no network comms for now
unset OMPI_MCA_btl_tcp_if_include 2>/dev/null || true
unset OMPI_MCA_oob_tcp_if_include 2>/dev/null || true

NP="${NP:-1}"
CONFIG="${CONFIG:-./configs/nano2.json}"

mkdir -p logs
ts="$(date +%Y%m%d-%H%M%S)"
log="logs/mpi-${ts}.log"

echo "[run_mpi] exec mpirun -np ${NP} ./build/nano2 --config ${CONFIG}"
exec mpirun -np "${NP}" ./build/nano2 --config "${CONFIG}" | tee "${log}"
