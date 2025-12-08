# nano2 (a nano version of nano-gpt)

A small Transformer from scratch trained on TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories) with byte-level tokens.
Single GPU first; same code also runs multi box with CUDA aware MPI.

Decoder
- Embed (byte, vector)
- +positional (sinusoidal added)
- x=x+Attn(LN(x)), which is (LN, attn, +res)
- x=x+FFN(LN(x)), which is (LN, FFN, +res)
- linear (tied to embedding), softmax, logits, xent

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
```
###Forward single box gpu:
```bash
mpirun -np 1 ./build/nano2_fw --config ./configs/nano2.json --fw-iters=1
mpirun -np 1 ./build/nano2_fw --config ./configs/nano2.json --fw-iters=5
mpirun -np 1 ./build/nano2_fw --config ./configs/nano2.json --fw-iters=10
```
time/iter: 442 ms

###Forward 2 boxes gpu: 
```bash
mpirun -np 2 --hostfile hostfile ./build/nano2_fw --config ./configs/nano2.json --fw-iters=1
mpirun -np 2 --hostfile hostfile ./build/nano2_fw --config ./configs/nano2.json --fw-iters=5
mpirun -np 2 --hostfile hostfile ./build/nano2_fw --config ./configs/nano2.json --fw-iters=10
```
time/iter: 926 ms

###Forward and Backward single box gpu:
```bash
mpirun -np 1 ./build/nano2_fwbw --config ./configs/nano2.json --fwbw-iters=1 
mpirun -np 1 ./build/nano2_fwbw --config ./configs/nano2.json --fwbw-iters=2
mpirun -np 1 ./build/nano2_fwbw --config ./configs/nano2.json --fwbw-iters=3
```

###Forward and Backward 2 boxes gpu:
```bash
mpirun -np 2 --hostfile hostfile --map-by ppr:1:node ./build/nano2_fwbw --config ./configs/nano2.json --fwbw-iters=1
mpirun -np 2 --hostfile hostfile --map-by ppr:1:node ./build/nano2_fwbw --config ./configs/nano2.json --fwbw-iters=2
mpirun -np 2 --hostfile hostfile --map-by ppr:1:node ./build/nano2_fwbw --config ./configs/nano2.json --fwbw-iters=3
```
12700 ms vs 15000 ms

###Forward single box CPU:
```bash
mpirun -np 1 ./build/nano2_cpu_fw --config ./configs/nano2.json --fw-iters=1
mpirun -np 1 ./build/nano2_cpu_fw --config ./configs/nano2.json --fw-iters=2
mpirun -np 1 ./build/nano2_cpu_fw --config ./configs/nano2.json --fw-iters=3
```
time/iter: 26700 ms

###Forward and Backward CPU:
```bash
mpirun -np 1 ./build/nano2_cpu_fwandbw --config ./configs/nano2.json --fw-bw-iters=1
mpirun -np 1 ./build/nano2_cpu_fwandbw --config ./configs/nano2.json --fw-bw-iters=2
mpirun -np 1 ./build/nano2_cpu_fwandbw --config ./configs/nano2.json --fw-bw-iters=3
```
time/iter: 166000 ms
