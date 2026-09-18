# FIDO Task 1 — r03 distance v2

`r03` conserva sin cambios la inferencia y la cabeza de keypoints usadas por
`r02-best`. Respecto a `r01-task1-keypoint`, mantiene la misma implementación
de inferencia que ya pasó la ingestión real de Codabench; respecto a `r02-best`,
el único cambio del bundle es reemplazar la cabeza de distancia por el
checkpoint `unet_dist_v2_0.6113.pth`.

## Pesos

| Archivo | MD5 |
|---|---|
| `checkpoints_from_pod/latest/keypoint_0.8537.pth` | `b935bc46dc794e46277f26a2e13139f0` |
| `checkpoints_from_pod/latest/unet_dist_0.5709.pth` | `aa8bcb00ed040cce99dd6b793468df16` |
| `checkpoints_from_pod/dist_v2/unet_dist_v2_0.6113.pth` | `863fb6c2eb350f4dd52897c5881019ea` |

## Evaluación local

Comando:

```text
python eval/run_local.py --submission submissions/r03-dist-v2 --task keypoints --save
```

Resultado sobre los 5 casos locales: `final_score=0.600000`,
`keypoint_auc=0.818182`, `distance_auc=0.090909`, con 0 timeouts.

## Advertencia sobre la métrica de distancia

El `val_distance_auc=0.6113` del checkpoint nuevo viene de una `evaluate()` que
descarta con `continue` los casos sin gap medible. El scorer oficial sí penaliza
esos casos, así que `0.6113` está inflado y no es comparable con el `0.0900`
obtenido en Codabench.
