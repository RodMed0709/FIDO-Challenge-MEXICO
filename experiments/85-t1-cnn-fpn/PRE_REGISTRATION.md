# T1-85 — ResNet-18 ImageNet + FPN stride 4

## Hipótesis

Un ResNet-18 con representación ImageNet y fusión top-down de `C2/C3/C4/C5`
producirá un heatmap de stride 4 más localizado que la CNN histórica de stride
16, manteniendo sin cambios evaluator, split por escenario, fold, semilla,
loss, decoder y denominador all-case.

## Diseño factorial limpio

- Control gemelo: `--backbone cnn --batch-size 2 --grad-accum-steps 4 --amp`.
- Candidato: `--backbone resnet18-fpn --pretrained-backbone
  --no-freeze-backbone --backbone-lr-multiplier 0.1`.
- Ambos se ejecutan en el mismo runtime con fold 0, seed 0, 20 épocas, mismo
  dataset/cache, microbatch 2, acumulación 4, AMP BF16, batch efectivo 8, LR de
  cabeza `1e-4`, loss y evaluación oficial 0..10.
- Pesos ImageNet se obtienen mediante `torchvision` y su cache estándar; no se
  descarga nada durante inferencia si ya están cacheados.
- Prohibido usar Mock Test para entrenar, elegir épocas o ajustar parámetros.

El efecto arquitectónico se estima únicamente como ResNet-FPN menos el control
CNN gemelo. La comparación histórica FP32 se reporta aparte y no se atribuye al
backbone porque mezcla arquitectura, precisión y régimen de batch.

## Expectativa y gate

GO si mejora `keypoint_auc >=0.005`, mejoran PCK@1 y PCK@3, y ningún escenario
cae más de 0.02 frente al control limpio. Si falla, se registra NO-GO y no se
mezcla con cambios de decoder/augmentación.

## Comando pendiente del pod

```bash
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --backbone cnn --fold 0 --seed 0 --epochs 20 --batch-size 2 \
  --grad-accum-steps 4 --amp --lr 1e-4 \
  --out checkpoints/task1_cnn_amp_accum_control

PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --backbone resnet18-fpn --pretrained-backbone --no-freeze-backbone \
  --backbone-lr-multiplier 0.1 --fold 0 --seed 0 --epochs 20 \
  --batch-size 2 --grad-accum-steps 4 --amp --lr 1e-4 \
  --out checkpoints/task1_resnet18_fpn
```

No ejecutar este entrenamiento durante la implementación local.
