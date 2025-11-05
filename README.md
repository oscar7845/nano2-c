# nano2 (a nano version of nano-gpt)

A small Transformer from scratch trained on TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories) with byte-level tokens.
Single GPU first; same code also runs multi box with CUDA aware MPI.

## Helpful commands 

```bash
# config + build (release)
./scripts/build.sh

# run single box
./scripts/run_local.sh

# run cluster of boxes
NP=2 ./scripts/run_mpi.sh

# multi node example (edit hostnames first)
./scripts/run_mpi.sh -H node1,node2 -N 2
