# T1-83 — DARK offline

**Fecha de pre-registro:** 2026-08-18  
**Estado:** preparado, no ejecutado  
**Control:** decoder global del checkpoint CNN limpio T1-80, fold 0  
**Único cambio:** corrección DARK sobre probabilidades sigmoid positivas

## Hipótesis y gate

La expansión de Taylor sobre el pico gaussiano reducirá la cuantización
subpíxel sin cambiar pesos, casos ni logits. GO únicamente si
`delta_auc >= 0.002`, suben PCK@1 y PCK@3, y ningún escenario pierde más de
`0.01` AUC. No se decide GO hasta producir el checkpoint/cache final limpio.

## Comando pendiente

```bash
PYTHONPATH=src python analysis/evaluate_task1_decoders.py \
  --logits checkpoints/task1_clean_cnn/fold0_logits.npz \
  --mode dark \
  --output experiments/83-t1-dark-decoder/results.json
```
