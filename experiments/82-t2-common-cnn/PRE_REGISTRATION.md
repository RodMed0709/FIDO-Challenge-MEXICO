# T2-82 — representación común CNN

## Hipótesis única

Dos encoders CNN separados, entrenados con correspondencias densas GT y una
pérdida InfoNCE con negativos intraimagen, aprenderán un espacio común entre
fundus y OCT en-face. Se reutiliza sin cambios la augmentación de escala exacta
y su valid mask aprobadas en T2-81.

## Datos y protocolo congelados

- Task 2 training únicamente; Mock Test prohibido para selección.
- GroupKFold por escenario, fold 0/5, seed 0.
- Estadísticas de normalización y rango de escala derivados exclusivamente de
  los índices train del fold y persistidos con sus índices fuente.
- Descriptor dim 128, K=64, temperatura 0.1, negativos intraimagen y radio de
  exclusión de 12 px. Sin pose head, DINO, hard negatives cross-frame ni una
  proyección OCT nueva.
- 10 épocas. Checkpoint, SHA-256, log JSONL y métricas finales persistidos.

## Comando pre-registrado

```bash
PYTHONPATH=src python -m fido.train.train_task2_common \
  --root /workspace/data/Task2 \
  --enface-cache /root/data_cache/Task2_enface \
  --encoder cnn --fold 0 --seed 0 --epochs 10 --k 64 \
  --out checkpoints/task2_common_cnn
```

## Gate

GO solamente si M1 top-1% >=50%, M2 con pose parcial GT >=50% a 10 px,
mediana M2 <37.12 px y OCT-shuffle reduce recall al menos 30 puntos. Si falla,
se cierra T2-82 y se abre T2-83; no se construye solver.
