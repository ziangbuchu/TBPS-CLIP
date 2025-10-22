#!/bin/bash

CUDA_VISIBLE_DEVICES="$1,$2,$3,$4"

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
main.py