#!/bin/bash
set -uo pipefail
cd /workspace/FIDO_CHALLENGE
source /workspace/venv/bin/activate
export PYTHONPATH=/workspace/FIDO_CHALLENGE/src

echo "=== ENTRENAMIENTO TASK 2 INICIADO $(date) ==="
python -m fido.train.train_task2_baseline \
  --root /workspace/data/Task2 --epochs 25 --batch-size 8 --lr 3e-4 \
  --n-folds 5 --fold 0 --out checkpoints/task2_real --val-every 5
echo "=== ENTRENAMIENTO TASK 2 TERMINADO $(date) (exit=$?) ==="

echo "=== ENTRENAMIENTO TASK 1 INICIADO $(date) ==="
python -m fido.train.train_task1_unet \
  --root /workspace/data/Task1 --epochs 15 --batch-size 8 --lr 1e-3 \
  --n-folds 5 --fold 0 --out /workspace/FIDO_CHALLENGE/checkpoints/task1_real
echo "=== ENTRENAMIENTO TASK 1 TERMINADO $(date) (exit=$?) ==="

echo "=== ORQUESTADOR (retry) COMPLETO $(date) ==="
