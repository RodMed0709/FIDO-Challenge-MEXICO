# 50 — Task 1, keypoint: ¿localiza la punta de la cánula sobre el fundus?

Peldaño de la escalera: **T1-R3** (`ATTACK_LADDER.md`).

## Escalera

| run | qué es | métrica |
|---|---|---|
| `../00-smoke-contract/` | control: centro de la imagen, sin modelo | `keypoint_auc` 0.0000 |
| smoke de overfit (5 casos, CPU) | ¿el arnés aprende algo? | `keypoint_auc` 0.145 |
| **50a** | CNN + heatmap sub-píxel, dataset completo, 20 épocas | **`val_keypoint_auc` 0.8534** (época 4, corrida aún viva) |

## Expectativa pre-registrada

El keypoint pesa **0.7** del score final de Task 1
(`final_score = 0.7·keypoint_auc + 0.3·distance_auc`, verificado en el scorer
congelado). El líder del leaderboard va en 0.619 de score final, así que un
`keypoint_auc` por encima de ~0.7 ya sería competitivo si la componente de
distancia acompaña.

El smoke test previo (5 casos, CPU local) llegó a 0.145 y bajó un ejemplo
individual de >150 px a 3.0 px en 60 pasos: el arnés aprende, la pregunta era si
escala al dataset completo.

## 🟢 RESULTADO PARCIAL (2026-08-17) — `val_keypoint_auc = 0.8534`

Época 4 de 20, fold 0 de 5, 14,399 casos de validación. Error medio **1.04 px**.

**La corrida seguía viva al escribir esto** (iba por la época 7). Los números
son del último corte de validación, no de una corrida terminada.

Es el número más alto que ha producido el proyecto, y llega en el componente que
más pesa. Contexto para juzgarlo: el paper de los organizadores
(arXiv 2603.25555), que resuelve esta misma task con el mismo simulador, reporta
7.93 px de error de keypoint.

**Costo**: compartido con el peldaño 10 en el mismo pod (~$1/h).

## Hallazgo del smoke test que sigue vigente

La norma de gradiente cruda explota (751 → 7469 en 40 pasos) y
`clip_grad_norm_` con `max_norm=1.0` domina casi todos los pasos: el clip actúa
como un learning rate fijo minúsculo. Por eso se expuso `--max-grad-norm`
(default 1.0). Si la convergencia se estanca, ahí está la primera palanca.

## ⚠️ Advertencias

1. **El checkpoint se sobrescribe cada vez que mejora el AUC.** Vive solo en el
   pod, que es efímero: `ledger/assets/ck-t1-keypoint.yaml`.
2. Aplica el defecto abierto `def-getitem-fallback-cruza-split`: el fallback de
   `__getitem__` salta sobre el dataset completo, no sobre el `Subset` del fold,
   así que la validación *puede* haber visto alguna muestra de train. Con
   14,399 casos el efecto esperado es pequeño, pero el número no es limpio hasta
   que se arregle.

## Cómo reproducir

```bash
python -m fido.train.train_task1_keypoint \
  --root /workspace/data/Task1 --epochs 20 --batch-size 8 --lr 1e-4 \
  --n-folds 5 --fold 0 --val-every 4 --case-cache /workspace/cache/task1_cases.json \
  --out /workspace/FIDO_CHALLENGE/checkpoints/task1_keypoint
```

Lanzado por `infra/orchestrate_task1.sh`. **`--case-cache` no es opcional**: sin
él cada lanzamiento reabre los 61,691 JSON del dataset (~15 min sobre MooseFS).

## Por qué existe este peldaño

El peldaño 10 midió la **distancia**, que pesa 0.3. Sin el keypoint, el score de
Task 1 tiene un techo de 0.3 por construcción. Este es el otro 0.7.

## Números

En `ledger/runs/run-2026-08-17-p50-t1-keypoint.yaml`, con su comando y su
procedencia. `python -m fido.ledger best --task 1` lo resume.
