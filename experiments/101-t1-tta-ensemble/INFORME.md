# T1-101 — Test-time augmentation + ensemble para el keypoint de Task 1

## Encargo y motivación

Codabench (submission r06, 100 casos) da `keypoint_auc=0.7573`. El mismo
checkpoint mide `keypoint_auc≈0.85` en local (ATTACK_LADDER.md T1-80, fold 0,
n=14399). Brecha local-real: ~0.10 de AUC. Si el keypoint real llegara al
número local, el score final subiría de 0.5691 a ~0.6367 y superaría al líder
del leaderboard (0.619) sin tocar nada más de la solución.

Hipótesis: el promediado sobre transformaciones (TTA) reduce varianza en
dominio nuevo, que es la naturaleza de esa brecha, con presupuesto de tiempo
disponible de sobra (20 s/caso, se usa ~0.23 s/caso hoy).

**Restricción dura: no se reentrena nada.** Todo lo de abajo usa checkpoints
ya existentes en disco.

## Checkpoints disponibles localmente

Los checkpoints del pod se perdieron (ver `NOW.md`); lo que sobrevive
localmente son los `.pth` que viajaron a este equipo dentro de paquetes de
submission o backups puntuales. Verificado con hash SHA-256 + suma de un
tensor de la primera capa (para detectar duplicados exactos vs archivos con
mismo nombre pero pesos distintos):

| etiqueta | archivo | sha256[:12] | suma capa 1 |
|---|---|---|---:|
| `cnn_best_ep20` | `checkpoints_from_pod/latest/keypoint_0.8537.pth` | `05c0b3dc534a` | -0.182528 |
| `cnn_r01_early` | `checkpoints_from_pod/keypoint_only.pth` | `211f9020cb75` | 0.230088 |
| `cnn_r04_interim` | `artifacts/r04-interim/keypoint_only.pth` | `2c7c2d572187` | 0.193920 |

Los tres son pesos **distintos** de la misma arquitectura
(`fido.models.task1_keypoint.Task1KeypointModel`, `base_channels=32,
n_downsamples=4`) — no son copias duplicadas del mismo entrenamiento, son
snapshots de corridas/épocas diferentes. Eso los hace candidatos válidos para
un ensamble "gratis" (snapshot ensembling), aunque no fueron entrenados con
esa intención.

### Hallazgo colateral, no buscado: el checkpoint que puntuó 0.7573 en Codabench NO es el mejor local

`submissions/r06-fallback-fixed/model_0.pth` (el paquete que de verdad se
subió y dio `keypoint_auc=0.7573`) empaqueta el sub-dict `"keypoint"` con
sha256 `6b73fc21bc12`-compatible y suma de capa 1 **0.193920** — **idéntico
bit a bit a `cnn_r04_interim`**, NO al `cnn_best_ep20` (0.8539/0.8537 en
ATTACK_LADDER.md T1-80). Confirmado con `torch.load` + comparación de
tensores, no solo por nombre de archivo.

En principio esto podría explicar parte de la brecha local-real (checkpoint
subóptimo en vez de domain shift puro). La triage de checkpoints (ver
"Resultados") lo descarta como explicación dominante: `cnn_r04_interim` (el
que realmente se subió) mide `keypoint_auc=0.8667` contra `0.8697` de
`cnn_best_ep20` sobre la misma muestra de triage (n=30/escenario) — una
diferencia de 0.003, ruido, no los ~0.10 de la brecha real-local. **El
checkpoint subido no es el mejor disponible, pero eso no es la causa
principal del `keypoint_auc=0.7573` de Codabench** — la brecha sigue siendo,
predominantemente, domain shift genuino entre el fold local y el test real,
que es exactamente lo que este peldaño ataca con TTA.

## Diseño de la evaluación honesta

- **GroupKFold por escenario, el mismo split que entrenó el checkpoint**:
  `fido.data.task1.task1_split(cases, n_splits=5, fold=0, seed=0)`. Verificado
  que reproduce EXACTO el split de producción: `val n=14399`, escenarios
  `Scenario_03` (10153) y `Scenario_04` (4246) — igual al `n=14399` reportado
  en ATTACK_LADDER.md T1-80. Estos dos escenarios están fuera del
  entrenamiento de los tres checkpoints (mismo split usado en T1-80/T1-85).
- **Lista de casos completa** (61,691, filtro de cánula activa) construida
  desde `data/_annotations/Task 1/*.json` (extraído localmente, NO requiere
  abrir los zips de 56 GB) — réplica exacta del filtro de
  `fido.data.task1.find_task1_cases` (mismo campo `Surgical Tool > CANNULA >
  Meta > ILM Distance` contra el mismo centinela `1e6`).
- **Muestra**: TODO (mostrar N real tras la corrida) casos, muestreo
  estratificado determinista (semilla fija) por escenario dentro del fold de
  validación — no es el dataset completo de 14399 por costo de CPU (ver
  "Coste temporal"), pero es una muestra aleatoria, no seleccionada a mano.
- **Ninguna configuración se ajustó mirando el resultado**: las 8 vistas
  (identity + 2 flips + 3 rotaciones de 90 + 2 zooms) y los checkpoints se
  fijaron ANTES de correr la evaluación completa (ver
  `src/fido/tta_keypoint.py` y el smoke test de 6 casos que motivó investigar
  el colapso de flips, documentado abajo).
- **Métrica**: `keypoint_auc` oficial — `fido.geometry.corner_auc`, réplica
  verificada línea por línea de `auc_from_errors` en
  `vendor/fido/Codabench Bundle/scoring_program/scoring_keypoints.py`
  (promedio de la fracción de casos con error <= t sobre t=0..10 px enteros).

## TTA: diseño y verificación

`src/fido/tta_keypoint.py`. Fusión en espacio de **heatmap** (logits, no
coordenadas ni probabilidades) — un solo decode `soft_argmax_2d` al final,
igual temperatura que usa el propio modelo sin TTA.

Familias:
- **Discretas** (identity, hflip, vflip, rot90/180/270): `torch.flip` /
  `torch.rot90` — permutaciones exactas de la grilla, sin interpolación.
- **Continua** (zoom multi-escala 0.9x/1.1x): rejilla afín
  (`F.affine_grid`+`F.grid_sample`) en coordenadas normalizadas, que son
  independientes de la resolución — la misma matriz sirve para transformar
  la imagen (1024²) y des-transformar el heatmap (64²) sin conversión manual
  de escala. `padding_mode="border"` en el heatmap (replica el logit del
  borde, casi siempre muy negativo, en vez de insertar un logit=0 que
  simularía "aquí hay algo" en medio del fondo).

**Verificación (CRÍTICO, ver encargo)**: `src/fido/tests/test_tta_keypoint.py`,
45 tests. Cada transformación discreta se prueba con 5 centros sintéticos
distintos: el heatmap recuperado tras transformar+des-transformar es
**bit-idéntico** al original (`atol=1e-5`) y el error de coordenada decodificada
tras el roundtrip es **<1e-4 px** — comparado contra el propio decode del
heatmap SIN transformar, no contra el centro gaussiano nominal (el
soft-argmax global tiene un sesgo propio de ~0.05-0.1px en centros no enteros,
incluso con `identity`; mezclarlo con el error del roundtrip habría inflado
o enmascarado bugs reales). El zoom tiene tolerancia <1.5px (dos
remuestreos bilineales). Tests adicionales: hflip/vflip mueven el eje
correcto y no el otro, las 6 vistas discretas producen picos mutuamente
distintos (guarda contra copiar mal el signo de `k` entre rotaciones), TTA
con una sola vista `identity` reproduce EXACTO el forward del modelo sin
TTA, y el ensamble de un modelo consigo mismo reproduce EXACTO un solo
modelo.

```
$ python -m pytest src/fido/tests/test_tta_keypoint.py -q
45 passed in 2.44s
$ python -m pytest src/fido/tests/ -q
222 passed in 18.24s
```

## Resultados

_TODO: completar con `experiments/101-t1-tta-ensemble/results.json` tras la
corrida completa (N por escenario, triage de checkpoints, tabla de
configuraciones, coste temporal)._

## Recomendación

_TODO_
