# T1-103 — Augmentacion geometrica+fotometrica del keypoint de Task 1

**Fecha de pre-registro:** 2026-08-19
**Estado:** preparado, NO ejecutado (smoke test en CPU si, entrenamiento real no)
**Semilla:** 0
**Split:** GroupKFold por escenario, 5 folds, fold 0 (mismo split que
T1-80/T1-85/T1-101: val = `Scenario_03` + `Scenario_04`, n=14399)

## Motivacion y encargo

Codabench (submission r06, 100 casos) da `keypoint_auc=0.7573`. El mismo
checkpoint mide `keypoint_auc=0.8697` en local sobre el fold 0 de validacion
(T1-80/T1-101). Brecha local-real de 0.10 AUC — si el keypoint real llegara al
nivel local, el score subiria de 0.5691 a ~0.6367 y superaria al lider del
leaderboard (0.619).

T1-101 midio, con 45 tests que descartan un bug de des-transformacion (ver
`src/fido/tests/test_tta_keypoint.py`, roundtrip <1e-4 px), que aplicar TTA
con flips y rotaciones de 90/180/270 DESTRUYE el checkpoint de produccion:
`keypoint_auc` 0.8697 -> 0.0061, error medio 70px
(`experiments/101-t1-tta-ensemble/results_small.json`,
config `ensemble_tta_combined`, aunque el colapso ya aparece con un solo
checkpoint — ver `baseline_no_tta` vs las corridas con vistas geometricas en
ese mismo archivo).

**Hipotesis de este peldano**: esa fragilidad es un problema de
ENTRENAMIENTO, no una propiedad inevitable del keypoint. El checkpoint actual
se entrena con `train_task1_keypoint.py` alimentando el fundus crudo a
`compute_loss` sin transformacion alguna — confirmado leyendo el loop de
entrenamiento completo (`src/fido/train/train_task1_keypoint.py`, funcion
`main`, lineas ~249-273): **cero augmentacion, geometrica o fotometrica**, hoy.
Un modelo que nunca vio una rotacion o un flip durante el entrenamiento no
tiene ninguna razon para ser equivariante a ellos. Y un modelo fragil a
transformaciones geometricas conocidas es plausiblemente fragil tambien al
domain shift local-real (que es, en esencia, una distribucion de apariencia
distinta) — esa es la conexion que este peldano pone a prueba.

## Que se compara

Arquitectura, split, loss, decoder, evaluador y receta de entrenamiento
IDENTICOS al baseline limpio de T1-80 (`--backbone cnn --epochs 20
--batch-size 8 --lr 1e-4 --n-folds 5 --fold 0 --seed 0`). El UNICO cambio
entre brazos es `--augment`:

| brazo | `--augment` | rol |
|---|---|---|
| control | `off` | reproduce T1-80 exacto (cero augmentacion) |
| candidato primario | `strong` | rotacion completa +-180 grados, escala 0.75-1.3x, traslacion +-15%, flip horizontal p=0.5, fotometricas fuertes (ver `src/fido/data/task1_augment.py::AUGMENT_PRESETS`) |
| candidato secundario (solo si `strong` degrada la metrica limpia) | `light` | rotacion +-15 grados, escala 0.9-1.1x, traslacion +-5%, mismas fotometricas atenuadas |

`strong` es el brazo primario porque cubre el mismo rango de vistas que
colapso en T1-101 (rotaciones de hasta 180 grados) — es la version mas directa
de la hipotesis. `light` queda pre-registrada como plan B, no como comparacion
obligatoria.

Ningun brazo usa el Mock Test para elegir hiperparametros ni para decidir el
gate (regla transversal de `ATTACK_LADDER.md`).

## Que NO cambia (controles congelados)

Arquitectura (`Task1KeypointModel`, `base_channels=32`, `n_downsamples=4`),
loss (`compute_loss`, L1 + BCE de heatmap sigma=2.0 peso=0.1), decoder
(`soft_argmax_2d` global), evaluador oficial (`fido.eval_task1`), split, fold,
semilla, LR, batch size, epocas, `--case-cache`. La augmentacion se aplica
SOLO al split de entrenamiento (`AugmentedTask1Dataset` envuelve el `Subset`
de train dentro de `train_task1_keypoint.py`) — validacion sigue viendo
fundus reales sin transformar, para que `keypoint_auc` de val siga siendo
comparable con T1-80/T1-85/T1-101 sin ningun cambio de metrica de por medio.

## Metricas

- **Primaria**: `keypoint_auc` oficial (`fido.eval_task1.evaluate_keypoints`,
  misma implementacion que T1-80/101) sobre el fold 0 de validacion, SIN TTA.
  Compara control vs candidato.
- **Secundaria (control directo de la hipotesis)**: tolerancia a TTA. Sobre el
  checkpoint candidato, correr el mismo TTA de 6 vistas discretas
  (`fido.tta_keypoint.default_flip_rotation_transforms`) que colapso el
  checkpoint de control en T1-101, y reportar `keypoint_auc_con_tta /
  keypoint_auc_sin_tta` para ambos checkpoints (control y candidato). Un valor
  cercano a 1.0 en el candidato (vs ~0.06 medido en el control en T1-101) es
  evidencia directa de que el modelo gano equivarianza geometrica real, no
  solo que "le fue mejor" por azar de inicializacion.
- Reportar tambien `mean_error` y desglose por escenario (`Scenario_03` /
  `Scenario_04` por separado, igual que T1-80/101) — un candidato que sube el
  AUC pooled pero empeora un escenario especifico no es una mejora limpia.

## Criterio GO / NO-GO

**GO** (adoptar `--augment strong` como receta de produccion para Task 1) si
se cumplen las tres condiciones:

1. `keypoint_auc` del candidato SIN TTA no cae mas de 0.02 por debajo del
   control (tolerancia de ruido de un solo entrenamiento; no se exige mejora,
   solo que la augmentacion no cueste generalizacion limpia).
2. La razon `keypoint_auc_con_tta / keypoint_auc_sin_tta` del candidato es
   `>= 0.7` (vs `~0.06` del control) — el colapso de T1-101 deja de ocurrir.
3. Ningun escenario individual (`Scenario_03`, `Scenario_04`) cae mas de 0.03
   de AUC frente al control.

**GO+** (evidencia mas fuerte que la minima, reportar aparte): el candidato
mejora el AUC limpio sobre el control en si mismo — indicaria que la
augmentacion tambien ataca directamente el domain shift local-real, no solo
la fragilidad a transformaciones sinteticas.

**NO-GO** si el candidato `strong` viola la condicion 1 o 3 (regresion de
generalizacion limpia) — en ese caso correr el brazo `light` (mismo criterio)
antes de descartar la hipotesis. Si `light` tambien falla la condicion 1/3
pero pasa la condicion 2 (mejora robustez a TTA sin sacrificar AUC limpio de
forma severa), registrar como GO parcial: usar augmentacion para TTA en
inferencia, pero mantener el checkpoint sin augmentar para el forward simple.
Si ninguna intensidad cumple la condicion 2, la hipotesis del encargo (la
fragilidad es un problema de entrenamiento) queda refutada para esta receta
de augmentacion — el siguiente sospechoso pasaria a ser la capacidad/
arquitectura del modelo, no la falta de augmentacion.

## Riesgos y limitaciones conocidas, declarados de antemano

- **`set_epoch` no se propaga a workers persistentes.** Con
  `--num-workers 8` (default de produccion) y `persistent_workers=True`, cada
  worker recibe una copia del dataset al arrancar; la llamada a
  `augmented_train_dataset.set_epoch(epoch)` en el proceso principal NO
  llega a esas copias (limitacion de `multiprocessing`, documentada en
  `AugmentedTask1Dataset.set_epoch`). Efecto practico: cada caso de train
  recibe UNA vista aumentada fija (determinada por `seed` + indice),
  reusada en las 20 epocas, no una vista nueva por epoca. Sigue siendo
  augmentacion real (el modelo ve rotaciones/flips/ruido que hoy no ve en
  absoluto) pero con menos diversidad de la ideal. Confirmado que
  `set_epoch` si funciona con `--num-workers 0` (usado en el smoke test).
  Si el resultado del pod es ambiguo, repetir con `--num-workers 0` (mas
  lento, I/O sin paralelizar) antes de descartar la hipotesis por esta
  limitacion.
- **El contraste (`apply_contrast`, `factor<1`) puede levantar ligeramente el
  vinetado por encima de negro puro.** Acotado deliberadamente
  (`contrast_range=(0.85,1.2)` incluso en `strong`) para que el efecto sea
  pequeno — verificado en `test_task1_augment.py::test_contrast_lift_on_vignette_periphery_is_bounded_for_preset_ranges`
  que un pixel de vinetado negro puro no sube de 0.06 bajo el factor minimo
  del preset. No es cero, es un compromiso documentado.
- **Reintento de muestreo con fallback a identidad**
  (`sample_geometric_transform`, `max_resample_attempts=8`): si un keypoint
  cerca del borde no encuentra una transformacion valida en 8 intentos, ese
  ejemplo se entrena SIN augmentar en esa epoca/indice particular. Con
  `keypoint_margin_px=8.0` y los rangos de `strong`, esto deberia ser raro
  (keypoints tipicamente centrados, ver `analysis/measure_task2_annotation_stats.py`
  para Task 2, no medido especificamente para Task 1 — riesgo abierto, no
  bloqueante).
- No se ha medido el costo temporal extra de la augmentacion (interpolacion
  bilineal + blur gaussiano ocasional por muestra) sobre 20 epocas reales.
  El smoke test de este documento confirma que corre, no que el overhead sea
  despreciable a escala completa.

## Smoke test en CPU (ejecutado, ver "Resultados" abajo)

Root sintetico (`Task1Dataset`-compatible, NO datos reales): 2 escenarios x 4
frames, fundus 1024x1024 con vinetado radial burdo + un punto brillante en el
keypoint real, JSON con `Ground Truth > Task 1` y cannula activa. Construido
por un script de scratchpad (fuera del repo), no forma parte del pre-registro
como dato de entrenamiento real.

## Comando exacto de lanzamiento en pod (NO ejecutado)

```bash
# Control -- reproduce T1-80 exacto
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --backbone cnn --epochs 20 --batch-size 8 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 --augment off \
  --out checkpoints/task1_t103_control

# Candidato primario
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --backbone cnn --epochs 20 --batch-size 8 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 --augment strong \
  --out checkpoints/task1_t103_strong

# (plan B, solo si strong viola condicion 1/3 del gate)
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --backbone cnn --epochs 20 --batch-size 8 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 --augment light \
  --out checkpoints/task1_t103_light
```

Tras cada corrida: evaluar TTA sobre el checkpoint resultante con
`fido.tta_keypoint.tta_predict_keypoint` + `default_flip_rotation_transforms()`
sobre el mismo val loader (fold 0), reportando la razon con/sin TTA descrita
arriba. Registrar en `ATTACK_LADDER.md` con comando exacto, commit y log,
igual que el resto de la escalera.

## Resultados

### Smoke test CPU (2026-08-19, root sintetico, `--device cpu --num-workers 0`)

```
$ PYTHONPATH=src python -m fido.train.train_task1_keypoint \
    --root <scratch>/smoke_root --epochs 3 --batch-size 2 --n-folds 5 \
    --num-workers 0 --val-every 1 --augment strong --device cpu \
    --out <scratch>/smoke_out_strong
AVISO: solo 2 escenarios únicos < n_folds=5 -> usando TODO el dataset para train y val.
[epoch 1] train_loss=57.5472 (coord_l1=57.3957 heatmap_bce=1.5146)
[epoch 1]   val_keypoint_auc=0.0000  val_mean_error=13.91px  n=8
  -> nuevo mejor, guardado en .../smoke_out_strong/keypoint_only.pth
[epoch 2] train_loss=7.7253 (coord_l1=7.3872 heatmap_bce=3.3802)
[epoch 2]   val_keypoint_auc=0.5227  val_mean_error=4.83px  n=8
  -> nuevo mejor
[epoch 3] train_loss=5.4870 (coord_l1=5.1982 heatmap_bce=2.8883)
[epoch 3]   val_keypoint_auc=0.3977  val_mean_error=6.08px  n=8
Mejor AUC: 0.5227 en época 2
```

Control (`--augment off`, mismo root/config) tambien corre y converge
(`train_loss` 61.8 -> 2.9 -> 4.0). Con `--num-workers 2` (spawn de Windows) la
corrida con `--augment light` tambien completa sin error — confirma que
`AugmentedTask1Dataset` es picklable para multiprocessing.

Checkpoint recargado independientemente con
`Task1KeypointModel(base_channels=32, n_downsamples=4).load_state_dict(...,
strict=True)` (42 tensores) y forward de prueba sobre un tensor aleatorio
1024x1024 — reload y forward correctos.

`python -m pytest src/fido/tests/ -q` -> **260 passed** (222 previos +
38 nuevos de `test_task1_augment.py`), sin regresiones.

Esto confirma que el pipeline CORRE, la loss baja, y el checkpoint
sobrevive guardado+recarga. No mide `keypoint_auc` real (el root es
sintetico, 8 casos, sin OCT ni anatomia real) — eso es exactamente lo que
las corridas de pod (arriba) deben producir.

### Corridas de pod

_Pendiente. No ejecutado en este peldano por instruccion explicita (no
entrenar de verdad)._
