# nano2 (a nano version of nano-gpt))

A small Transformer from scratch trained on TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories) with byte-level tokens.
Single GPU first; same code runs multi box with CUDA aware MPI.

## Helpful commands 

```bash
# Configure + build (Release)
./scripts/build.sh

# Run locally (single process)
./scripts/run_local.sh

# Run with MPI on one box (2 ranks)
NP=2 ./scripts/run_mpi.sh

# Multi-node example (edit hostnames first)
./scripts/run_mpi.sh -H node1,node2 -N 2
