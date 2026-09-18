# T2-81 — augmentación exacta de escala

## Hipótesis única

Aplicar al fundus una escala geométrica alrededor del centro y componer el GT
exactamente como `M_aug = H @ M`, con rango logarítmico derivado exclusivamente
de `train_idx`, mejora el error de escala fuera del soporte robusto del fold sin
fuga de validación. No se cambian modelo, loss, optimizador ni datos.

## Evidencia que abre el rung

T2-80 fold 0 real: baseline AUC `0.004765`, mediana `41.451 px`; sustitución de
escala GT AUC `0.018695`, mediana `37.118 px`. Posición sigue siendo el mayor
cuello de botella; este rung aísla solamente escala.

## Corrida congelada

- Task 2 train, GroupKFold 5, fold 0, seed 0. Nunca Mock.
- Cuantiles 5/95 de `log(scale)` calculados solo sobre `train_idx`; banda
  simétrica multiplicativa expandida 25%.
- Único cambio: `--scale-augment`; validación no recibe transform.
- El valid mask excluye padding de correspondencias posteriores.

```bash
PYTHONPATH=src python analysis/evaluate_task2_scale_holdout.py \
  --root /workspace/data/Task2 --fold 0 --seed 0 \
  --enface-cache /root/data_cache/Task2_enface \
  --baseline-checkpoint checkpoints/task2_fixed/model_1.pth \
  --augmented-checkpoint checkpoints/task2_scale_aug/model_1.pth \
  --output experiments/81-t2-scale-augmentation/holdout_results.json

PYTHONPATH=src python -m fido.train.train_task2_baseline \
  --root /workspace/data/Task2 --enface-cache /root/data_cache/Task2_enface \
  --fold 0 --seed 0 --scale-augment --out checkpoints/task2_scale_aug
```

## Gate

GO únicamente si sube `corner_auc`, baja al menos 20% el error de escala del
holdout sintético y ningún escenario pierde más de `0.01` AUC. Si falla, cerrar
la rama sin tuning.
