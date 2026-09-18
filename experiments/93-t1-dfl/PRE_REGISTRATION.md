# T1-93 — cabeza de distancia por bins (Distribution Focal Loss)

**Fecha:** 2026-08-19
**Estado:** pre-registrado, listo para lanzar en pod (RTX 5090)
**Datos:** exclusivamente Task 1 train (`data/Task 1`, `data/_annotations/Task 1`);
Mock Test prohibido para elegir hiperparámetros (`CONSTITUTION.md` §I.3)
**Split:** GroupKFold por escenario, `n_folds=5`, `fold=0`, `seed=0`
(`fido.data.task1.task1_split` — obligatorio, mismo criterio que
`train_task1_unet.py`/`train_task1_keypoint.py`)

---

## 0. El número que justifica todo esto (medido antes de escribir código)

**Pregunta:** ¿en qué fracción de casos reales `distance_from_segmentation`
(la función que usa `submissions/r06-fallback-fixed/inference.py` en
producción, línea ~228) devuelve NaN — es decir, no encuentra clase
instrumento o clase ILM en la segmentación?

Dos mediciones independientes, ambas sobre datos LOCALES (no Mock Test):

### 0.1 Techo estructural (segmentación PERFECTA — máscaras GT, no un checkpoint)

`analysis/measure_t1_distance_rescale_effect.py --stride 1`, sobre los
**61,691 casos** con cánula activa de los 10 escenarios completos
(`data/_annotations/Task 1` + `data/_bscan_seg/Task 1`):

| | n | fracción |
|---|---:|---:|
| Cánula activa, **SIN medición geométrica posible** (ni con máscara GT) | 15,577 | **25.25%** |
| Cánula activa, con medición geométrica | 46,114 | 74.75% |

Es decir: **incluso con segmentación perfecta, uno de cada cuatro casos NO
tiene clase instrumento y clase ILM simultáneamente visibles en el mismo
B-scan.** No es un problema de que el modelo segmente mal — es estructural:
el instrumento o el ILM sencillamente no están ambos en el B-scan en ese
frame. Ningún refinamiento del `UNet` de segmentación puede arreglar esto.

Bono de la misma corrida — impacto en AUC oficial (umbral 0..20, target
`GT/7.8125`) de aplicar el fallback a los no medibles, con segmentación
PERFECTA (techo teórico del enfoque geométrico):

- Solo casos medibles (n=46,114): `distance_auc = 0.7459`.
- Todos los casos con cánula activa, fallback=159.2px sin reescalar para
  los no medibles (n=61,691): `distance_auc = 0.5575`.

Esto acota lo que CUALQUIER pipeline geométrico puede lograr como máximo:
**~0.56 de AUC oficial incluso con segmentación perfecta**, muy por encima
de nuestro `distance_auc≈0.09` real en Codabench. La brecha 0.09→0.56 no se
explica solo por la cobertura — hay más señal perdida en algún punto entre
"segmentación real del checkpoint" y "Codabench" que este rung no resuelve
(ver §6, riesgos).

### 0.2 Fracción real con el checkpoint de producción (`submissions/r06-fallback-fixed/model_0.pth`)

`analysis/measure_task1_segmentation_nan_fraction.py`, muestra sistemática
por escenario, B-scans leídos EN MEMORIA directo de los `.zip` (sin
extraer a disco, 56GB no caben en los ~88GB libres de `D:`):

Dos corridas independientes sobre Scenario_01 (6,703 casos con cánula
activa), con tamaños de muestra distintos, dan el mismo resultado -- y una
tercera corrida (Scenario_02) muestra que la fracción VARÍA sustancialmente
entre escenarios, así que ninguna corrida de un solo escenario es
representativa del dataset completo:

| corrida | n muestreado | n medido | n NaN | fracción NaN |
|---|---:|---:|---:|---:|
| Scenario_01, muestra sistemática n=300 | 300 | 198 | 102 | **34.0%** |
| Scenario_01, muestra sistemática n=60 (repetición independiente) | 60 | 39 | 21 | **35.0%** |
| Scenario_02, muestra sistemática n=60 | 60 | 54 | 6 | **10.0%** |
| **Pooled (Scenario_01 n=60 + Scenario_02 n=60)** | **120** | **93** | **27** | **22.5%** |

**La corrida completa sobre los 10 escenarios sigue en background en el
momento de escribir esto** (I/O de leer los `.zip` de 56GB en memoria,
~2 min/escenario incluso con el batching optimizado — ver
`experiments/93-t1-dfl/NAN_FRACTION_MEASUREMENT.md`, que esta misma corrida
sobreescribe automáticamente al terminar con la tabla completa por
escenario). Se reporta este resultado parcial YA, sin esperar la corrida
completa, porque el punto 0 pre-registrado exige reportar el número real
"aunque el resto falle". El pooled de los dos escenarios medidos hasta
ahora (22.5%, n=120) está por debajo del techo estructural medido con
segmentación perfecta sobre TODO el dataset (25.25%, §0.1, n=61,691) --
pero Scenario_01 por sí solo (34-35%) ya lo supera. La variación
10%-35% entre escenarios es real (probablemente distinta visibilidad del
instrumento/ILM según el tipo de cirugía o el ángulo de cada escena) y es
justo la razón por la que NO se reporta un único número "definitivo" de un
escenario aislado -- se actualiza este documento cuando termine la corrida
completa de 10 escenarios.

**Conclusión del punto 0:** la sospecha se confirma. El fallback constante
no es un caso raro -- es el resultado en **entre 1 de cada 10 y 1 de cada 3
casos reales** con OCT presente, dependiendo del escenario (10.0%-35.0%
medido, pooled parcial 22.5% sobre 2 de 10 escenarios), y en **1 de cada 4
casos incluso con segmentación PERFECTA** (25.25%, dataset completo,
n=61,691, §0.1). Cualquiera de estos números por sí solo ya justifica
construir una cabeza que prediga SIEMPRE, sin depender de que la
segmentación encuentre nada.

---

## 1. Hipótesis

Una cabeza DFL (regresión directa desde el B-scan a una distribución sobre
bins de distancia, sin pasar por segmentación intermedia) predice en el
100% de los casos con OCT presente, y por lo tanto:

1. En el subconjunto donde la segmentación **SÍ mide** hoy, DFL es
   comparable o algo peor que el pipeline geométrico (la geometría tiene
   ventaja: usa la anotación de segmentación pixel-a-pixel como señal
   intermedia, con `R²=0.9916` en el ajuste lineal gap→distancia sobre esos
   casos).
2. En el subconjunto donde la segmentación **NO mide** hoy (cae al
   fallback constante `203.8px`), DFL gana con margen — porque el fallback
   no usa ninguna información del caso, y DFL sí.
3. El `distance_auc` oficial **all-case** (comparable directo con nuestro
   `0.09` real de Codabench) sube frente al baseline geométrico actual en
   el mismo split, sin necesariamente alcanzar el techo teórico de §0.1
   (`0.56`, que asume segmentación perfecta — DFL parte de una arquitectura
   mucho más simple que el YOLO-NAS+cross-attention del paper base).

## 2. Diseño de la cabeza (`src/fido/models/task1_distance_dfl.py`)

- **Encoder**: CNN convolucional (`GroupNorm`, mismo bloque que
  `unet_bscan_seg._ConvBlock`) aplicado con **pesos compartidos** a cada una
  de las 2 B-scans ortogonales de un caso (igual criterio que la
  segmentación: "de forma independiente a cada uno de los 2 B-scans").
  Global-average-pool por slice, concatenación de los 2 vectores resultantes
  (preserva la identidad de cada B-scan en vez de promediarla a ciegas).
- **Cabeza DFL**: dos bloques `Linear+BatchNorm1d+ReLU` + `Linear` final a
  `reg_max+1` logits. Adaptación explícita de "dos bloques conv (1x1, 3x3) +
  BN/ReLU + conv 1x1 final" del paper (`INFORME.md` §2.2) — el paper opera
  sobre mapas de features espaciales de un detector denso (anclas por
  posición); nuestro problema es un escalar por caso, así que la cabeza
  corre sobre el vector ya pooled, no sobre un mapa espacial.
- **Decodificación**: softmax sobre los `reg_max+1` bins → esperanza
  ponderada → reescalado lineal a `[d_min, d_max]`
  (`ŷ = d_min + E[bin]/reg_max·(d_max-d_min)`).
- **Pérdida**: Distribution Focal Loss (Li et al. 2020, GFLv2) — cross-entropy
  interpolada entre los dos bins que rodean el target continuo, en vez de
  contra un único bin discreto.
- **Transferencia opcional**: `load_pretrained_segmentation_encoder` copia
  `encoders.*` desde el `UNet` de segmentación ya entrenado (mismas formas
  exactas si `base_channels`/`depth` coinciden) — inicialización, no
  obligatoria para que el entrenamiento funcione.
- **Robustez a target fuera de rango**: `encode_target` recorta a
  `[0, reg_max]` antes de construir la pérdida — un target fuera de
  `[d_min, d_max]` nunca produce NaN/Inf, solo satura el gradiente hacia el
  bin extremo (verificado en tests, ver §4).

## 3. Rango `[d_min, d_max]` y `reg_max` — elegidos por medición, no copiados del paper

El paper usa `d_min=-1mm, d_max=6mm` porque uno de sus dos tipos de cirugía
(inyección subretiniana) tiene distancias negativas (punta dentro del
tejido) y su sensor/unidades son distintos a los nuestros. Medimos la
distribución REAL del target oficial (`gt[2]/7.8125`, `gt =
annotation["Ground Truth"]["Task 1"]`) sobre los 61,691 casos con cánula
activa
(`analysis/measure_task1_distance_target_histogram.py`, resultado completo
en `experiments/93-t1-dfl/DISTANCE_TARGET_HISTOGRAM.md`):

| | valor |
|---|---:|
| n | 61,691 |
| min | 6.704px |
| max | 1178.769px |
| media | 254.429px |
| mediana (p50) | 216.692px |
| std | 166.611px |
| casos negativos | **0 (0.00%)** |
| p0.1 | 24.750px |
| p1 | 42.041px |
| p99 | 812.033px |
| p99.9 | 1131.277px |

**Diferencia real frente al paper:** en nuestros datos NO existe régimen
negativo (a diferencia de la inyección subretiniana del paper) — la punta
de la cánula nunca está "dentro" del tejido según esta anotación. Elegimos:

- `d_min = 0` (ningún caso observado por debajo; simplifica la
  interpretación de los bins sin perder cobertura real).
- `d_max = 1200` (por encima del máximo observado, 1178.8px — deja margen
  sin recortar casi ningún caso real del propio dataset de train).
- `reg_max = 128` → ancho de bin = `1200/128 = 9.375px`, más fino que el
  umbral oficial de distancia (enteros 0..20px) para que la esperanza
  ponderada tenga margen de resolución sub-bin. El costo de subir `reg_max`
  es trivial (una capa `Linear` final de 129 salidas).

Nota sobre por qué la resolución en la cola extrema (>p99) importa poco: el
score oficial (`corner_auc`/`auc_from_errors`) solo premia error `≤20px` —
un acierto fallado por 50px o por 900px cuenta igual (cero en todos los
umbrales). Lo que importa es tener bins finos donde el modelo puede acertar
cerca del valor real, no resolución fina en la cola.

## 4. Tests (`src/fido/tests/test_task1_distance_dfl.py`) — PASANDO en CPU

```
PYTHONPATH=src python -m pytest src/fido/tests/test_task1_distance_dfl.py -q
```

Salida real:

```
....................                                                     [100%]
20 passed in 4.90s
```

Cubren: round-trip encode→decode (~0 de error, identidad algebraica exacta
de la distribución "dos-calientes"), la pérdida DFL bajando en un caso
sintético (Adam, 60 pasos, `loss[-1] < 0.5·loss[0]`), targets fuera de rango
(`±1e4`) sin NaN/Inf ni excepción, validación de hiperparámetros inválidos,
forma de salida del modelo completo con entrada en cero (caso sin OCT), y
transferencia de pesos del encoder de segmentación (copia exacta cuando las
formas calzan, 0 copiados sin romper cuando no calzan).

## 5. Smoke test de entrenamiento en CPU — PASANDO

Sobre una muestra local de 5 escenarios × 15 frames extraídos en memoria de
los `.zip` (sin GPU, `torch==2.13.0+cpu`):

```
PYTHONPATH=src python -m fido.train.train_task1_distance_dfl \
  --root <root_extraido_5_escenarios> --epochs 2 --batch-size 4 --device cpu \
  --num-workers 0 --n-folds 5 --fold 0 \
  --base-channels 8 --depth 2 --reg-max 16 --d-min -20 --d-max 320 \
  --out <checkpoint_dir>
```

Salida real:

```
Fallback (mediana train, casos sin OCT): 360.574px n_fit=60
[epoch 1] train_dfl_loss=2.8253
[epoch 1]   val_distance_mae=323.41px  val_distance_auc=0.0000  coverage_oct=1.0000  n=40
  -> nuevo mejor (auc=0.0000 mae=323.41px) en epoch 1, checkpoint guardado
[epoch 2] train_dfl_loss=2.4796
[epoch 2]   val_distance_mae=318.56px  val_distance_auc=0.0000  coverage_oct=1.0000  n=40
  -> nuevo mejor (auc=0.0000 mae=318.56px) en epoch 2, checkpoint guardado
Mejor AUC: 0.0000 en época 2
```

`loss` baja de forma monótona (2.8253→2.4796) en 2 épocas sobre 60 casos de
train — esperado con tan pocos datos/épocas, NO es una corrida válida para
elegir hiperparámetros ni para reportar como resultado. Confirma que el
pipeline completo (dataset → split → modelo → loss → checkpoint) corre sin
romper en CPU.

`analysis/evaluate_task1_distance_dfl.py` también se corrió end-to-end sobre
el mismo checkpoint de smoke test — sin errores, incluyendo el caso borde de
un subconjunto con `n=0` (esta muestra de 5 escenarios × 15 frames, tomados
del inicio de cada video, no tuvo NINGÚN caso con segmentación medible —
artefacto de la extracción, no señal real).

## 6. Comando exacto de lanzamiento (pod, RTX 5090)

```bash
cd /workspace/FIDO_CHALLENGE
PYTHONPATH=src /workspace/venv/bin/python -m fido.train.train_task1_distance_dfl \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --epochs 50 --batch-size 16 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 --device cuda \
  --reg-max 128 --d-min 0 --d-max 1200 \
  --base-channels 32 --depth 4 --head-hidden 128 \
  --pretrained-segmentation-checkpoint submissions/r06-fallback-fixed/model_0.pth \
  --out checkpoints/task1_distance_dfl \
  2>&1 | tee experiments/93-t1-dfl/train.log

PYTHONPATH=src /workspace/venv/bin/python analysis/evaluate_task1_distance_dfl.py \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --dfl-checkpoint checkpoints/task1_distance_dfl/model_0.pth \
  --segmentation-checkpoint submissions/r06-fallback-fixed/model_0.pth \
  --n-folds 5 --fold 0 --seed 0 --device cuda \
  --output experiments/93-t1-dfl/EVALUATION.md \
  2>&1 | tee experiments/93-t1-dfl/eval.log
```

`--pretrained-segmentation-checkpoint` es opcional (transferencia de pesos
del encoder ya entrenado por `train_task1_unet.py`) — lanzar también sin él
si el tiempo de pod lo permite, para aislar si la transferencia ayuda.

## 7. Métricas y gate — PRE-REGISTRADOS (no se tocan tras ver resultados)

Todas sobre GroupKFold fold 0, `analysis/evaluate_task1_distance_dfl.py`,
umbral oficial de distancia (enteros **0..20**, `MAX_THRESHOLD_DIST`, no el
`MAX_THRESHOLD_PX=10` de keypoints — ver nota de bug en §8):

- `distance_auc_pooled_dfl` vs `distance_auc_pooled_geometrico` (mismo split,
  mismos casos).
- `distance_auc_no_medible_dfl` vs `distance_auc_no_medible_geometrico`
  (subconjunto donde `distance_from_segmentation` da NaN con el checkpoint
  de segmentación real — el corazón de la hipótesis).
- `distance_auc_medible_dfl` vs `distance_auc_medible_geometrico`.

**GO** (promover DFL como candidato para ensamblar/reemplazar el fallback en
producción): `distance_auc_no_medible_dfl > distance_auc_no_medible_geometrico`
**Y** `distance_auc_pooled_dfl >= distance_auc_pooled_geometrico`.

**NO-GO** (descartar o rediseñar): DFL pierde en el subconjunto no-medible,
o pierde pooled por un margen que no compensa ninguna ganancia en el
subconjunto no-medible.

**Zona intermedia** (gana en no-medible pero pierde pooled, o viceversa):
no promover sin una comparación explícita de ensamble (usar geométrico
cuando mide, DFL cuando no) contra ambos por separado — ese es el siguiente
rung, no este.

## 8. Hallazgo colateral — bug de umbral en `fido.eval_task1.evaluate_distances`

`src/fido/eval_task1.py` importa `MAX_THRESHOLD_PX` (=10) de
`fido/geometry.py` y lo usa como umbral **tanto para keypoints como para
distancia** dentro de `_summarize`/`evaluate_distances`. El umbral oficial
de distancia es **20**, no 10
(`vendor/fido/Codabench Bundle/scoring_program/scoring_keypoints.py`:
`MAX_THRESHOLD_PX=10` para keypoint, `MAX_THRESHOLD_DIST=20` para
distancia, líneas 10-11 y 262-268). Esto significa que **todo
`val_distance_auc` histórico calculado con `evaluate_distances`**
(`train_task1_unet.py`, `analysis/measure_task1_distance_ceiling.py`) usa un
umbral más angosto que el real — subestima el AUC oficial verdadero (menos
umbrales acumulados en el promedio, todos con accuracy ≤ la que tendrían
con el umbral correcto, porque la CDF de error es no decreciente).

Esta cabeza DFL **no reutiliza esa función** — `official_distance_auc` en
`train_task1_distance_dfl.py` y `evaluate_task1_distance_dfl.py`
implementan el umbral 0..20 directamente, citando la constante oficial. No
se modifica `eval_task1.py` en este rung (afecta a otros scripts en uso;
corregirlo es una decisión aparte, fuera del alcance de T1-93) — se deja
registrado aquí para que quien lo toque después no lo repita.

## 9. Riesgos / lo que este rung NO resuelve

- La brecha entre el techo teórico de segmentación perfecta (`§0.1`, `0.56`
  all-case) y el score real de Codabench (`0.09`) es mucho mayor que lo que
  explica solo la cobertura — sugiere que hay señal perdida en algún otro
  punto (calibración específica de Codabench, normalización, o algo aún no
  identificado) que este rung no ataca ni descarta.
- El encoder es deliberadamente más simple que el YOLO-NAS+cross-attention
  del paper (sin fusión con OPMI, sin multi-escala) — si DFL no gana, no
  necesariamente refuta el enfoque DFL en general, podría ser capacidad de
  encoder insuficiente.
- `d_max=1200` cubre el 99.9%+ de train pero el test oculto de Codabench
  podría tener una cola más larga — `encode_target` lo maneja sin romper
  (satura al bin extremo), pero la precisión ahí se degrada por diseño.
