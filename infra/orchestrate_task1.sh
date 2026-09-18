#!/bin/bash
# Orquestador de Task 1 tras la auditoria de rendimiento del 2026-08-17.
#
# Orden deliberado: keypoint ANTES que distancia. La distancia ya tiene un
# modelo real medido (AUC 0.5681, respaldado en /workspace/checkpoints_backup/),
# mientras que el keypoint nunca se entreno sobre datos reales -- y es el
# componente que mas pesa en el score de Task 1.
#
# Todas las banderas de abajo son fixes con causa medida, no tuning:
#   --case-cache      : evita reabrir los 61,691 JSON en cada lanzamiento (15 min -> 0.5 s)
#   --class-weights   : pesos ya medidos sobre el dataset completo, evita otro escaneo total
#   --val-every       : la validacion sobre MooseFS cuesta mas que las epocas de train
# Ademas, Task1Dataset ahora carga SOLO lo que cada script consume (ver su docstring).
set -uo pipefail
cd /workspace/FIDO_CHALLENGE
source /workspace/venv/bin/activate
export PYTHONPATH=/workspace/FIDO_CHALLENGE/src
export PYTHONUNBUFFERED=1   # sin esto el log se queda mudo por buffering y parece colgado

CACHE=/workspace/cache/task1_cases.json

echo "=== T1-R3 KEYPOINT INICIADO $(date) ==="
python -m fido.train.train_task1_keypoint \
  --root /workspace/data/Task1 --epochs 20 --batch-size 8 --lr 1e-4 \
  --n-folds 5 --fold 0 --val-every 4 --case-cache "$CACHE" \
  --out /workspace/FIDO_CHALLENGE/checkpoints/task1_keypoint
echo "=== T1-R3 KEYPOINT TERMINADO $(date) (exit=$?) ==="

echo "=== T1-R5 DISTANCIA INICIADO $(date) ==="
python -m fido.train.train_task1_unet \
  --root /workspace/data/Task1 --epochs 15 --batch-size 8 --lr 1e-3 \
  --n-folds 5 --fold 0 --val-every 3 --case-cache "$CACHE" \
  --class-weights 0.0341661,0.4904578,2.4753761 \
  --out /workspace/FIDO_CHALLENGE/checkpoints/task1_real
echo "=== T1-R5 DISTANCIA TERMINADO $(date) (exit=$?) ==="

echo "=== ORQUESTADOR TASK 1 COMPLETO $(date) ==="
