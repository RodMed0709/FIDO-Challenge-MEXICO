# T1-81 — techo real de distancia

**Fecha:** 2026-08-18  
**Estado:** pre-registrado, no ejecutado  
**Datos:** exclusivamente Task 1 train; Mock Test prohibido  
**Split:** GroupKFold 5, fold 0, semilla 0

## Hipótesis

`376.7 px = 0.7320 × 511 + 2.6779` es un límite algebraico, no un techo
empírico: el rango GT `[5.2, 308]` puede estar completamente debajo de él y aun
así haber casos sin punta/ILM visibles en ninguna de las dos máscaras. El techo
honesto es el AUC all-case de la geometría aplicada a máscaras GT, usando para
los no medibles un fallback ajustado solo con el train fold.

La calibración (`scale`, `offset`, `fallback`) debe venir del artefacto
`distance_calibration.json` que acompaña al checkpoint y el conjunto de sus
`fit_case_ids` debe ser exactamente igual al conjunto de IDs del train fold,
sin duplicados, faltantes ni extras. Cualquier discrepancia aborta la corrida.
El reporte incluye SHA-256 del checkpoint y del sidecar, además de metadata de
fold/seed/época/commit embebida en el checkpoint cuando exista.

## Métricas y gate congelados

- `oracle_distance_auc_all`, `distance_auc_all` y coverage, pooled y por escenario.
- Todos los casos de validation permanecen en el denominador.
- Si el techo oráculo es `<0.25`, priorizar cabeza aprendida.
- Si es `>=0.35`, conservar geometría como candidato.
- Entre ambos valores, no promover sin comparación directa en T1-89.
- No se modifica checkpoint, calibración ni hiperparámetros tras esta escritura.

## Artefacto esperado

La corrida escribe `CEILING.md` con comando, hash del checkpoint, provenance de
calibración, métricas y decisión. Este archivo no contiene resultados anticipados.

```bash
cd /workspace/FIDO_CHALLENGE
PYTHONPATH=src /workspace/venv/bin/python analysis/measure_task1_distance_ceiling.py \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --checkpoint /workspace/FIDO_CHALLENGE/checkpoints/task1_dist_v2/model_0.pth \
  --calibration /workspace/FIDO_CHALLENGE/checkpoints/task1_dist_v2/distance_calibration.json \
  --repo /workspace/FIDO_CHALLENGE --device cuda --batch-size 8 \
  --n-folds 5 --fold 0 --seed 0 \
  --output experiments/81-t1-distance-ceiling/CEILING.md \
  2>&1 | tee experiments/81-t1-distance-ceiling/run.log
```
