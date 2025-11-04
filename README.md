# nano2 (C + CUDA + MPI)

A tiny, from-scratch Transformer trained on TinyStories with byte-level tokens.
Single GPU first; same code runs multi-process with CUDA-aware MPI.

## Quick start

```bash
# Configure + build (Release)
./scripts/build.sh

# Run locally (single process)
./scripts/run_local.sh

# Run with MPI on one box (2 ranks)
NP=2 ./scripts/run_mpi.sh

# Multi-node example (edit hostnames first)
./scripts/run_mpi.sh -H node1,node2 -N 2
