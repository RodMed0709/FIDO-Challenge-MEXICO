# T1-82 — local soft-argmax offline

**Fecha de pre-registro:** 2026-08-18  
**Estado:** preparado, no ejecutado  
**Control:** decoder global del checkpoint CNN limpio T1-80, fold 0  
**Único cambio:** decoder local, ventana 7

## Hipótesis y gate

Restringir el soft-argmax alrededor del máximo principal eliminará la influencia
de modos remotos sin cambiar pesos, casos ni logits. GO únicamente si
`delta_auc >= 0.002`, suben PCK@1 y PCK@3, y ningún escenario pierde más de
`0.01` AUC. No se decide GO hasta producir el checkpoint/cache final limpio.

## Comando pendiente

```bash
PYTHONPATH=src python analysis/evaluate_task1_decoders.py \
  --checkpoint checkpoints/task1_clean_cnn/keypoint_only.pth \
  --cache-logits checkpoints/task1_clean_cnn/fold0_logits.npz \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --mode global \
  --output experiments/82-t1-local-decoder/global_control.json

PYTHONPATH=src python analysis/evaluate_task1_decoders.py \
  --logits checkpoints/task1_clean_cnn/fold0_logits.npz \
  --mode local --window 7 \
  --output experiments/82-t1-local-decoder/results.json
```
