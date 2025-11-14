# nano2 (a nano version of nano-gpt)

A small Transformer from scratch trained on TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories) with byte-level tokens.
Single GPU first; same code also runs multi box with CUDA aware MPI.

Decoder
Embed (byte, vector)
+positional (sinusoidal added)
x=x+Attn(LN(x)), which is (LN, attn, +res)
x=x+FFN(LN(x)), which is (LN, FFN, +res)
linear (tied to embedding), softmax, logits, xent

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
