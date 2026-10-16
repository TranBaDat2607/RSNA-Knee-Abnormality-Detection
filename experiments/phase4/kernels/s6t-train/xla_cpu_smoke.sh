#!/bin/bash
# local XLA (PJRT CPU) smoke test of the S6T TPU patch: 2 epochs with a stop after epoch 0, then resume
set -e
V=${SCRATCH:?set SCRATCH}/xlavenv/bin/python
export PJRT_DEVICE=CPU S6T_ALL_DATA=1 S6T_PAD_W=8 OMP_NUM_THREADS=4
ARGS="--px ${SCRATCH:?set SCRATCH}/devjpg/px_s0 --targets targets_dev.csv --out out --arms E_prior --backbone convnext_small --init timm --seeds 1 --seed-base 7 --folds 1 --epochs 2 --batch 2 --train-windows 8 --crop 96 --slices 3 --phys-mm 3.5 --spatial-attn --grad-ckpt --lr 1e-4 --head-lr 1e-3 --wd 0.05 --workers 1 --eval-workers 1 --aug crop --train-sampling uniform --eval-stride 3 --resume --ckpt-every 1 --i-have-user-authorisation-to-finetune"
rm -rf out
$V -u ${SCRATCH:?set SCRATCH}/compact_train_s6t_xla.py $ARGS --stop-after-epoch 0 || echo "first run exit $?"
ls -la out/ckpt
$V -u ${SCRATCH:?set SCRATCH}/compact_train_s6t_xla.py $ARGS
ls -la out/models
