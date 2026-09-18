# T1-80 — clean all-case baselines

**Fecha de pre-registro:** 2026-08-18  
**Estado:** preparado, no ejecutado  
**Semilla:** 0  
**Split:** GroupKFold por escenario, 5 folds, screening en fold 0

## Hipótesis

El fallback recursivo histórico de `Task1Dataset.__getitem__` sustituía un caso
ilegible por el siguiente índice global. Esa sustitución podía cruzar el
`Subset` de entrenamiento/validación, cambiar el denominador e introducir fuga
entre escenarios. Reintentar exclusivamente el mismo caso y fallar con su
`scenario/frame_id` hará la medición reproducible y all-case.

Los valores históricos CNN `0.8537`, DINOv2 frozen `0.8005` y distancia
`0.6113` no son baselines limpios. No se compararán como si fueran resultados
del split corregido. En distancia, la calibración afín y el fallback (mediana)
se ajustarán exclusivamente con IDs del train fold; todo caso no medible
conservará su lugar en el denominador. El `159.2 px` histórico solo se conserva
como default de compatibilidad del evaluador, no como valor del gate limpio.

## Controles congelados

No se cambia arquitectura, decoder, loss, resolución ni augmentations. Se
reproducirán únicamente:

1. CNN histórica, 20 épocas, batch 8, LR `1e-4`.
2. DINOv2 frozen histórico, 20 épocas, batch 4, LR `1e-4`.

## Expectativa y gate

- El conteo evaluado debe ser exactamente el tamaño del fold.
- Train y validación deben tener cero escenarios compartidos.
- Los IDs usados para ajustar calibración/fallback deben ser únicos y disjuntos
  de validación; valores extremos de val no pueden cambiar esas constantes.
- Coverage de distancia se reportará aparte y el AUC usará todos los casos.
- Se espera que el AUC de distancia limpio sea menor que `0.6113`; no se fija
  un delta mínimo porque este rung sanea medición, no selecciona arquitectura.
- Si una lectura sigue fallando tras tres intentos totales, la corrida falla de
  forma explícita; nunca se sustituye el caso.
- `--class-weights` históricos requieren marcar la corrida explícitamente como
  transductiva y no son admisibles para decidir el gate limpio.

## Comandos preparados (no ejecutados localmente)

```bash
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --backbone cnn --epochs 20 --batch-size 8 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 \
  --out checkpoints/task1_clean_cnn

PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --backbone dinov2 --freeze-backbone --epochs 20 --batch-size 4 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 \
  --out checkpoints/task1_clean_dino_frozen
```

Cada corrida deberá guardar comando exacto, commit, log, predicción por caso y
métricas pooled/per-scenario antes de cerrar el rung en `ATTACK_LADDER.md`.
