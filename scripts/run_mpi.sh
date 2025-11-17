#!/usr/bin/env bash
set -euo pipefail

#cd to repo root (robust to symlinks)
cd "$(dirname "${BASH_SOURCE[0]}")/.."

#MPI on a single node
export OMPI_MCA_btl=self,vader
unset OMPI_MCA_btl_tcp_if_include 2>/dev/null || true
unset OMPI_MCA_oob_tcp_if_include 2>/dev/null || true

#(off) pick a GPU for this run
#usage: GPU=0 ./scripts/run_local.sh
if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

#config path (override with CONFIG=/path.json)
CONFIG="${CONFIG:-./configs/nano2.json}"

#build if missing
if [[ ! -x ./build/nano2 ]]; then
  echo "[run_local] nano2 missing; building..."
  ./scripts/build.sh
fi

#log
mkdir -p logs
ts="$(date +%Y%m%d-%H%M%S)"
log="logs/local-${ts}.log"

echo "[run_local] exec ./build/nano2 --config ${CONFIG}"
#use 'exec' so signals (Ctrl-C) propagate cleanly; tee for a local log.
exec ./build/nano2 --config "${CONFIG}" | tee "${log}"

