# r04 interim joint

Submission conjunta de corte intermedio para medir ambas tareas con un solo
snapshot reproducible.

- Task 1 keypoint: CNN limpio, fold 0/5, seed 0, mejor checkpoint disponible
  en epoch 5 (`val_keypoint_auc=0.8539`, `val_mean_error=1.02 px`, n=14399).
- Task 1 distance: cabeza UNet conservada de `r03-dist-v2`; no se atribuye como
  mejora nueva.
- Task 2: baseline CNN entrenado con augmentacion geometrica exacta de escala,
  fold 0/5, seed 0, mejor checkpoint epoch 38.

La carpeta incluye siempre ambos pesos porque el ingestion oficial selecciona
`model_0.pth` o `model_1.pth` segun la tarea.

## Verificacion local

```powershell
python eval/run_local.py --submission submissions/r04-interim-joint --task both --save
```

Resultados sobre los 5 casos de Mock Test, sin seleccionar hiperparametros con
sus etiquetas:

- Task 1: `0.625455` (`keypoint_auc=0.854545`,
  `distance_auc=0.090909`), 0 timeouts.
- Task 2: `0.000000`, 0 timeouts. La mejora del holdout de escala no se
  transfiere al Mock fuera de distribucion y no se considera un avance de
  leaderboard.

Task 2 requirio `KMP_DUPLICATE_LIB_OK=TRUE` solamente para ejecutar el scorer
vendorizado en Windows, por un choque local entre dos copias de
`libiomp5md.dll`. Ingestion e inferencia terminaron normalmente antes del choque;
el rerun produjo el score anterior.

SHA-256:

- `inference.py`: `63FBEEC0DAD456C539DBA7984208F74A29330922FDCBA211D32DBCCEA09E256F`
- `model_0.pth`: `6B73FC21BC16BA1AE004E35E875298036C88EA6D777303ABCF63015CBAA773DA`
- `model_1.pth`: `C365077A30533BBA5BEF9874A78A1F1FC11E89471939600F231AAE9C4310B01D`
