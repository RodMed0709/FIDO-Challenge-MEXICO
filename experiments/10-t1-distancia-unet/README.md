# 10 — Task 1, distancia: ¿se puede MEDIR en vez de regresar a ciegas?

Peldaño de la escalera: **T1-R5** (`ATTACK_LADDER.md`).

## Escalera

| run | qué es | métrica |
|---|---|---|
| `../00-smoke-contract/` | control: sin modelo | `distance_auc` 0.0000 |
| piso trivial | la mejor constante posible (159.2 px) | `distance_auc` 0.0445 |
| **10a** | UNet de segmentación + medición geométrica | **`val_distance_auc` 0.5681** |

## Expectativa pre-registrada

`distance_auc > 0.15` ya sería una mejora sustancial sobre el piso de 0.0445.
Escrito antes de correr, en `ATTACK_LADDER.md` T1-R5.

## 🟢 RESULTADO (2026-08-16/17) — `val_distance_auc = 0.5681`

**12.8x el piso trivial**, muy por encima de la expectativa. Medido en la época 2
sobre 12,566 casos de validación, fold 0 de 5 (GroupKFold por escenario).

La idea: NO regresar el escalar a ciegas. En su lugar segmentar la cánula y la
ILM en el B-scan y **medir** la separación, apoyándose en la relación ya
verificada en T1-R2 sobre 92,012 mediciones reales:

```
distancia_GT = 0.7320 * pixel_gap + 2.6779     (R² = 0.9916)
```

**Costo**: ~$5 de pod (varias corridas fallidas por I/O antes de que entrara).

## Estado del checkpoint

`model_0.pth` (época 2) está respaldado en el pod:
`/workspace/checkpoints_backup/model_0_auc0.5681_ep2.pth`

**No está en `runs/` de este repo todavía** — el pod es efímero, hay que bajarlo.

## ⚠️ Advertencias para quien retome esto

1. **La corrida murió en la época 3** por `OSError: [Errno 6]` del filesystem de
   red (MooseFS), no por un bug de código. El 0.5681 es de la época 2, no de una
   corrida completa de 15 épocas. Entrenar más probablemente mejora.
2. **El AUC reportado está medido sobre un subconjunto favorable.** `evaluate()`
   descarta con `continue` los casos sin OCT y aquellos donde la segmentación no
   produce un gap medible (`NaN`). El scorer oficial NO descarta nada: mete un
   `PENALTY_ERROR` y lo cuenta en el denominador. Una auditoría midió que el
   **9.03%** de los casos tiene `dist_GT > 376.7 px`, que es el techo estructural
   del método con un B-scan de 512 filas — esos casos son inalcanzables por
   construcción y hoy se están descartando en vez de penalizar.
   **El AUC real de Codabench será menor que 0.5681.** Corregir antes de
   confiar en el número para decidir una submission.
3. Para el fallback sin OCT, la mejor constante medida sobre los 61,691 casos es
   **159.2 px** (`distance_auc` 0.0445), no 50.0 como usa la submission de humo.

## Cómo reproducir

```bash
python -m fido.train.train_task1_unet \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --class-weights 0.0341661,0.4904578,2.4753761 \
  --epochs 15 --batch-size 8 --lr 1e-3 --n-folds 5 --fold 0 --val-every 3 \
  --out checkpoints/task1_real
```

**Usa `--root /root/data_cache/Task1` (disco local), no `/workspace/data/Task1`**
(MooseFS): 9.1 ms/caso contra 108 ms/caso. Ver `HANDOFF.md`.
