#!/usr/bin/env bash
set -euo pipefail

#cd to repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

#config MPI

#tot # ranks (1 GPU per box, so default 1)
NP="${NP:-1}"

#either HOSTS="node1,node2" or HOSTFILE="hosts.txt"
HOSTS="${HOSTS:-}"
HOSTFILE="${HOSTFILE:-}"

unset OMPI_MCA_btl_tcp_if_include 2>/dev/null || true
unset OMPI_MCA_oob_tcp_if_include 2>/dev/null || true
unset OMPI_MCA_pmix_ptl_base_if_include 2>/dev/null || true

#config path (override with CONFIG=/path.json)
CONFIG="${CONFIG:-./configs/nano2.json}"

#build if missing
if [[ ! -x ./build/nano2 ]]; then
  echo "[run_mpi] nano2 missing; building..."
  ./scripts/build.sh
fi

#log
mkdir -p logs
ts="$(date +%Y%m%d-%H%M%S)"
log="logs/mpi-${ts}.log"

#construct mpirun command
mpi_cmd=(mpirun -np "${NP}")
if [[ -n "${HOSTS}" ]]; then
  mpi_cmd+=(-H "${HOSTS}")
fi
if [[ -n "${HOSTFILE}" ]]; then
  mpi_cmd+=(-hostfile "${HOSTFILE}")
fi

echo "[run_mpi] exec ${mpi_cmd[*]} ./build/nano2 --config ${CONFIG}"

exec "${mpi_cmd[@]}" ./build/nano2 --config "${CONFIG}" | tee "${log}"

