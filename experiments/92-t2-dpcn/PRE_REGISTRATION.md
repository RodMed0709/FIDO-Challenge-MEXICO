# T2-92 — DPCN/DPCN++: correlación de fase log-polar diferenciable

## Contexto que cambia el criterio de éxito (2026-08-19)

El leaderboard público de Codabench (`phases/27554/get_leaderboard`, task_id
33629) muestra **siete equipos por encima de cero** en Task 2: cnielsen
**0.475**, Alex 0.376, oli4more 0.28, kwonmj 0.279, dancies 0.244,
tianmaxingkong 0.145, akkkkk 0.111 — contra un techo teórico de **0.6389**
con matriz GT perfecta (T2-80). Esto refuta el veredicto interno previo de
"Task 2 sin señal explotable" (cinco cierres: apariencia T2-70/83, vasos
T2-20/88, DPCN razonado-no-medido, instrumento T2-89, priors). Ver
`PLAN.md` sección "CORRECCIÓN CRÍTICA" y `RESULTS.md` (2026-08-19).

**Esto NO es un motivo para no medir DPCN — es lo contrario: ahora sabemos
que hay señal real que capturar, así que esta corrida deja de ser "probar
algo que probablemente no funciona" y pasa a ser "perseguir una señal ya
confirmada por terceros".** Pero también cambia qué cuenta como éxito: el
listón ya no es "AUC > 0", es acercarse al rango donde compite el resto del
campo.

**Aviso importante, verificado parcialmente durante esta implementación**
(detalle completo al final de este documento, sección "Hallazgo paralelo"):
el script standalone del oráculo de apariencia
(`analysis/measure_task2_oracle_signal.py`) tiene por DEFECTO la convención
en-face NULA (`identity__keep_u__keep_v`), no la validada
(`transpose__flip_u__flip_v`), y todo indica que
`experiments/70-t2-oracle/ORACLE_FULL.md` (1214 casos, la base de la
memoria `fido-task2-appearance-oracle-negative.md`) corrió con ese defecto
sin corregir. Repetido en vivo sobre 83 casos reales (Scenario_03, recién
extraído) con ambas convenciones: el signo de la correlación media
(NCC de la proyección `mean`) SÍ cambia (z-score -0.22 con la convención
mala -> +0.12 con la correcta), pero la magnitud sigue siendo débil/no
significativa a este tamaño de muestra — no es la prueba contundente que
sugería la comparación de humo de 5 casos. **Conclusión calibrada: el bug
de convención en el oráculo es real y probablemente distorsionó su
resultado, pero no basta por sí solo para explicar un AUC de 0.475 en el
leaderboard; sigue siendo necesario re-correr el oráculo completo (1214
casos) con la convención corregida para saber si hay señal de apariencia
real** (accionable en CPU, sin pod). **Alcance importante: este bug NO
afecta a ningún modelo entrenado** (`Task2Dataset`, usado por
`task2_baseline.py`, `task2_common.py`, `task2_dinov2.py`, y este
`task2_dpcn.py`, usa `TASK2_ENFACE_CONVENTION` correcta desde
`fido/data/common.py`, no el default de este script de diagnóstico) — no
explica por qué los modelos ya entrenados dieron AUC=0.00, solo pone en
duda el veredicto de "no hay correlación de apariencia" como argumento
para NO intentar métodos basados en apariencia/correlación. DPCN no
depende de este resultado (aprende su propia geometría de punta a punta),
así que sigue siendo válido medirlo independientemente de cómo se resuelva
esto.

## Hipótesis

Una red DPCN/DPCN++ (Zhao et al. 2019/2021: correlación de fase
diferenciable en dominio log-polar, que desacopla rotación/escala de
traslación) sobre features aprendidas por encoders densos y separados por
modalidad, con un esquema de dos etapas (localización gruesa por
correlación cruzada + registración fina Fourier-Mellin dentro de una
ventana candidata recortada), recupera la similitud reflejada de 4 DOF de
Task 2 muy por encima de AUC=0.00.

Ataca explícitamente el problema central que el propio paper de DPCN++
documenta como su punto débil (sección 6.4: degradación con solapamiento
pequeño) mediante la ventana candidata: la correlación de fase nunca ve el
fundus completo (1024x1024, huella real ~2.4% del área), solo un recorte
generoso alrededor de una localización gruesa aprendida.

## Qué se construyó

- `src/fido/models/task2_dpcn.py`: encoders densos por modalidad SIN pesos
  compartidos (`DenseModalityEncoder`, 3-4 etapas, 48-384 canales según la
  rama), transformada log-polar diferenciable (`build_log_polar_grid` +
  `log_polar_warp`, grid fijo vía `grid_sample`), correlación de fase
  diferenciable vía espectro de potencia cruzada normalizado
  (`phase_correlation`, FFT real, soft-argmax local reutilizando
  `heatmap_decode.local_soft_argmax_2d`), desambiguación de 180 grados
  (ambigüedad real de Fourier-Mellin clásico sobre espectros de magnitud,
  no inventada: `resolve_rotation`), localización gruesa por correlación
  cruzada estilo SiamFC + recorte diferenciable de ventana candidata
  (`batched_crop_resize`), y una cabeza de 4 esquinas (parametrización de
  DeTone, alineada con la métrica oficial) con ajuste cerrado a similitud
  propia por pseudoinversa fija (`fit_similarity_from_corners`) e
  inicialización en cero (el modelo arranca siendo exactamente el
  estimador clásico Fourier-Mellin, sin perturbación aleatoria).
- `src/fido/train/train_task2_dpcn.py`: entrenamiento con GroupKFold por
  escenario (obligatorio, reutiliza `select_group_fold` de
  `train_task2_common.py`), augmentación de escala exacta (T2-81,
  `derive_scale_augmentation`/`Task2GeometricAugment`, derivada solo de
  train), y **control OCT-shuffle en cada validación** (baraja el en-face
  dentro del batch, fundus y GT intactos; reporta AUC y error mediano con y
  sin barajar, y su delta, en la misma línea de log).
- `src/fido/tests/test_task2_dpcn.py`: 9 tests, todos con casos sintéticos
  de respuesta conocida. Ver "Verificación" abajo.

### Por qué NO se usa `fit_reflected_similarity`

Documentado en detalle en el docstring del módulo. Resumen: el pipeline de
datos de Task 2 (`Task2Dataset`) ya compone el GT nativo (reflejado) con la
matriz fija de la convención en-face (`canonicalize_task2_matrix`, det=-1),
así que en el marco canónico donde opera todo el código de Task 2
existente (`task2_baseline.py`, `task2_common.py`, y ahora `task2_dpcn.py`)
la matriz objetivo es una similitud PROPIA, no reflejada. Imponer la
estructura de `fit_reflected_similarity` (para matrices reflejadas) a datos
que en este marco no lo están introduciría el mismo bug de ambigüedad de
signo que se quería evitar, con el signo cambiado. `fit_similarity_from_corners`
implementa el ajuste lineal análogo con la estructura correcta para este
marco, verificado exacto (`test_fit_similarity_from_corners_recovers_known_transform_exactly`).

## Verificación (smoke tests, CPU, PASANDO)

```
$ PYTHONPATH=src python -m pytest src/fido/tests/test_task2_dpcn.py -v
...
9 passed in 21.54s
```

Cobertura:
1. `phase_correlation` recupera un desplazamiento de píxeles exacto.
2. El log-polar convierte una rotación+escala sintética conocida en un
   desplazamiento recuperable (< 0.03 rad, < 0.02x de error).
3. `fit_similarity_from_corners` recupera exactamente (sin residuo, atol
   1e-3) la matriz de una similitud propia dados sus 4 corners, y es
   diferenciable.
4. **Pipeline Fourier-Mellin completo** (log-polar + correlación de fase +
   desambiguación de 180°, sin ninguna red de por medio) recupera 4
   transformaciones sintéticas fijas y verificadas con error de esquina
   < 8px (observado: 1.35-4.15px).
5. `apply_similarity_warp` y `similarity_matrix_about_center` (las dos
   representaciones de una misma transformación) son consistentes.
6. `Task2DPCNModel` completo produce las formas esperadas, TODOS los
   parámetros reciben gradiente (verificado nombre por nombre), y la
   cabeza de refinamiento de esquinas (inicializada en cero) no perturba
   la estimación clásica al arrancar.
7. Validación de forma de `valid_mask`.

**Hallazgo real durante la implementación, no un bug de este código**: el
núcleo Fourier-Mellin clásico SIN ENTRENAR, sobre contenido sintético
arbitrario (barrido aleatorio de ~15-20 combinaciones theta/escala/
traslación/textura), falla con error de esquina >>10px en aproximadamente
la mitad de los casos aleatorios — incluso sin ruido de medición, con
contenido perfectamente conocido en ambos lados. Causa verificada
(instrumentada, no adivinada): para esas combinaciones, el pico GLOBAL de
la superficie de correlación de fase en el dominio log-polar no coincide
con la ubicación de la transformación real; hay picos espurios más altos.
Es una limitación conocida de Fourier-Mellin clásico sobre espectros de
magnitud de bajo contenido discriminante, documentada en la literatura de
registración clásica, no un bug de signo o convención (los tests 1-3 y 5
descartan eso con precisión exacta). Es precisamente la motivación de (a)
correlacionar sobre FEATURES APRENDIDAS por los encoders densos, no sobre
el espectro de magnitud del píxel crudo, y (b) la cabeza de refinamiento de
esquinas. El entrenamiento real es quien decide si los encoders aprenden a
hacer el espectro más discriminante; los tests solo prueban que la
maquinaria clásica es matemáticamente correcta cuando SÍ encuentra el pico
correcto (que es la mayoría, no todos, de los casos sintéticos probados).
Los 4 casos usados en el test 4 son deterministas y verificados uno a uno,
elegidos para probar la matemática, no para afirmar robustez universal del
núcleo sin entrenar sobre contenido adversarial.

## Datos y protocolo

- Task 2 training únicamente para selección de hiperparámetros y
  arquitectura; Mock Test prohibido para eso (`CONSTITUTION.md` §3).
- GroupKFold por escenario (10 escenarios), fold 0/5, seed 0.
- Augmentación de escala exacta (T2-81): rango derivado SOLO de los
  índices de train del fold, con extrapolación 0.25 (cubre el rango de
  Mock/test, que no se solapa con train: train [126.66, 188.09]px, Mock
  [213.81, 226.16]px).
- `--window-size 512` (px, fundus 1024): cubre con margen holgado (~2.3-
  2.7x) el rango de escala aumentado.
- Control OCT-shuffle en CADA validación, reportado junto al AUC principal
  (no opcional, no al final: en cada línea de log de validación).

## Criterio GO/NO-GO (recalibrado con el leaderboard real)

Antes de esta corrección, el listón era "AUC > 0". Con el leaderboard real
a la vista (rango 0.111-0.475, mejor 0.475, techo 0.6389), ese listón ya no
distingue señal real de ruido de entrenamiento — un modelo puede arañar un
AUC positivo minúsculo sin capturar nada generalizable, exactamente el
riesgo que ya se vio en T2-R2/T2-R3 (memorización sin señal). El criterio
se recalibra así:

**GO** (vale la pena invertir más cómputo en esta familia, iterar
hiperparámetros, o intentarlo en la Final Round) si TODAS se cumplen:
1. `val_corner_auc >= 0.10` en el fold de validación (GroupKFold, no Mock).
   Referencia: el equipo más débil del leaderboard que puntúa por encima de
   cero está en 0.111 -- este listón dice "al menos competitivo con el peor
   equipo que resolvió algo", no "mejor que cero".
2. **El control OCT-shuffle degrada el AUC al menos 30 puntos porcentuales**
   relativos (`(auc_normal - auc_shuffle) / auc_normal >= 0.30`), o el AUC
   con shuffle cae a un intervalo consistente con el piso trivial. Sin
   esto, un AUC positivo no distingue "aprendió a usar el OCT" de "aprendió
   un prior de posición/escala del fundus que no necesita el OCT en
   absoluto" (el modo de fallo exacto de T2-R2).
3. `val_mean_error_px` mejora monótonamente durante al menos las primeras
   10 épocas (no memorización inmediata seguida de estancamiento, el patrón
   que cerró T1-80/85 para el backbone de Task 1).

**NO-GO** (cerrar esta corrida específica, no necesariamente la familia
DPCN completa -- ver nota) si CUALQUIERA:
- `val_corner_auc < 0.02` tras 30 épocas (indistinguible de ruido/piso).
- El control OCT-shuffle NO degrada el AUC (delta < 10 puntos): el modelo
  no está usando el OCT, el número es basura sin importar cuán alto se vea.
- Colapso de gradiente/NaN no recuperable en las primeras 5 épocas.

**Nota sobre alcance del NO-GO**: un NO-GO de ESTA corrida (hiperparámetros
concretos: `window_size=512`, `working_size=256`, `fine_feat_size=128`,
`radius_bins=96`, `angle_bins=192`) no cierra la familia DPCN entera --
dado que ahora sabemos que hay señal real explotable (leaderboard), un
NO-GO aquí es evidencia para ajustar hiperparámetros (ventana más
generosa, más resolución en el núcleo Fourier-Mellin, más peso a la
pérdida auxiliar de localización gruesa) antes de descartar la familia,
a diferencia del estándar previo a esta corrección donde un NO-GO
razonablemente cerraba la vía.

## Comando de entrenamiento (pod, RTX 5090, cu128)

```bash
cd /workspace/FIDO_CHALLENGE
PYTHONPATH=src python -m fido.train.train_task2_dpcn \
  --root /workspace/data/Task2 \
  --enface-cache /root/data_cache/Task2_enface \
  --epochs 40 --batch-size 4 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 \
  --window-size 512 --working-size 256 --fine-feat-size 128 \
  --scale-augment --scale-extrapolation 0.25 \
  --oct-shuffle-check \
  --out checkpoints/task2_dpcn
```

Smoke test primero (5 casos, overfit, confirma que el arnés real no tiene
bugs estructurales antes de gastar las 40 épocas completas):

```bash
PYTHONPATH=src python -m fido.train.train_task2_dpcn \
  --root "data/Mock Test/Task 2" \
  --epochs 20 --batch-size 2 --n-folds 1 --oct-shuffle-check \
  --out checkpoints/task2_dpcn_smoke
```
(con `--n-folds` menor que el número de escenarios disponibles, el script
usa train=val completo automáticamente y avisa que solo vale para overfit,
mismo mecanismo que `train_task2_baseline.py`.)

## Riesgos / supuestos abiertos

- El tamaño de ventana (`--window-size 512`) y la resolución del núcleo
  Fourier-Mellin (`--fine-feat-size 128`) son las primeras palancas a mover
  si el AUC se queda cerca de cero pero el control OCT-shuffle SÍ degrada
  (señal real pero débil): una ventana más generosa o más resolución en
  `radius_bins`/`angle_bins` (hoy 96/192, hardcoded en `LogPolarConfig` por
  defecto de `fine_feat_size`) son ajustes baratos antes de rediseñar.
- La pérdida auxiliar de localización gruesa (`coarse_center_loss`, peso 5x
  fijo) no se barrió -- si la etapa gruesa no converge a un centro
  razonable, la ventana recortada nunca contendrá la huella real y la
  etapa fina no tiene nada que encontrar. Vigilar `coarse_center` en los
  logs de las primeras épocas.
- No se implementó reanudación de entrenamiento (resume/checkpoint
  intermedio) como sí tiene `train_task2_common.py` para DINOv2 -- si el
  pod se cae a media corrida hay que relanzar desde cero. Aceptado como
  alcance reducido para esta primera medición; portar
  `ValidMicrobatchAccumulator`/`atomic_torch_save` es una mejora barata si
  esta corrida resulta prometedora y se justifica una corrida más larga.

## Hallazgo paralelo: posible bug de convención en el oráculo de apariencia

Durante esta implementación se auditó `analysis/measure_task2_oracle_signal.py`
(el script detrás de `experiments/70-t2-oracle/ORACLE_FULL.md`, la medición
de 1214 casos que sostiene "ninguna proyección del OCT correlaciona con el
fundus", memoria `fido-task2-appearance-oracle-negative.md`) buscando los
sospechosos que pidió Rodrigo tras la corrección del leaderboard
(convención en-face, forma de proyectar, sistema de coordenadas,
parametrización de salida).

**Hecho verificado**: el script tiene `DEFAULT_CONVENTION =
"identity__keep_u__keep_v"` (la convención NULA -- sin transponer ni
voltear) como valor por defecto de `--enface-convention`, mientras que la
convención validada y usada por todo el resto del pipeline de datos de
Task 2 es `transpose__flip_u__flip_v` (`TASK2_ENFACE_CONVENTION` en
`fido/data/common.py`). Ambas Tablas A (landmarks de vasos) y B (NCC/MI de
proyecciones) de ese script dependen directamente de este parámetro
(`hit_rates(...,convention,...)`, `warp_fundus(...,convention)`); el
"autotest de crosshair" que sí reporta 0.0000px de error NO depende de la
convención (solo valida la matriz GT contra las anotaciones del crosshair
del JSON), así que pasar ese autotest no garantiza que la convención usada
para las Tablas A/B sea la correcta.

**Evidencia circunstancial de que `ORACLE_FULL.md` corrió con el defecto
sin corregir**: su encabezado NO incluye los campos `en-face convention` ni
`radii mode` que sí aparecen en `experiments/70-t2-oracle/SMOKE_default.md`
y `SMOKE_fixed.md` (mismo directorio de experimento, comparación A/B a
n=5 casos que sí existe y sí distingue explícitamente "default" de
"fixed"). Los tres archivos (`SMOKE.md`, `SMOKE_default.md`,
`SMOKE_fixed.md`) y `ORACLE_FULL.md` llegaron al repo en el mismo commit
squash (`967dfba`), así que `git log` no puede fechar el orden real; el
formato del encabezado es la única pista disponible, y apunta a que
`ORACLE_FULL.md` es anterior a la corrección.

**Verificación propia, en vivo, sobre datos reales** (no solo el n=5 del
smoke test): se extrajo `data/Task 2/Scenario_03.zip` (83 casos reales,
antes solo disponible como zip sin extraer localmente) y se corrió
`measure_task2_oracle_signal.py` dos veces, una por convención
(`experiments/70-t2-oracle/RECHECK_scenario03_default.md` y
`RECHECK_scenario03_fixed.md`):

| convención | proyección `mean`: GT NCC | z-score |
|---|---:|---:|
| `identity__keep_u__keep_v` (defecto, probable bug) | -0.0752 | -0.2155 |
| `transpose__flip_u__flip_v` (validada) | +0.0174 | +0.1193 |

El signo cambia (igual dirección que el smoke test de 5 casos, que iba de
-0.36 a +0.23), pero la magnitud a n=83 sigue siendo débil/no significativa
-- muy lejos de la señal fuerte que explicaría un AUC de 0.475 en el
leaderboard. La Tabla A (landmarks de vasos) tampoco muestra una mejora
limpia con la convención corregida (ratios observado/nulo mixtos, algunos
por debajo de 1 incluso con la convención correcta).

**Conclusión, calibrada, no sobre-vendida**: el bug de convención en este
script de diagnóstico es real y probablemente sesgó `ORACLE_FULL.md`, pero
la re-medición en un escenario real (n=83) no lo confirma como la
explicación completa de "no hay señal de apariencia" -- sigue siendo
ambiguo a este tamaño de muestra. **Acción recomendada para el
orquestador/Rodrigo, no ejecutada aquí por alcance**: extraer los 10 zips
de `data/Task 2/` (27.3 GB, ya están en local, no hace falta pod) y
re-correr `measure_task2_oracle_signal.py --enface-convention
transpose__flip_u__flip_v --root data/Task2 --output <archivo>` sobre los
1214 casos completos -- es un diagnóstico de CPU de posiblemente menos de
una hora, mucho más barato que una corrida de DPCN en GPU, y su resultado
(señal clara vs. seguir sin señal a n=1214) debería informar si vale la
pena una siguiente iteración de DPCN con una cabeza de apariencia
explícita, o si el camino ganador de otros equipos es puramente geométrico/
aprendido (reforzando el diseño actual de DPCN, que no depende de
correlación de apariencia cruda).

Como nota de alcance: este bug, aun si se confirma al 100% con los 1214
casos, **no afecta a ningún modelo ya entrenado** de Task 2 -- `Task2Dataset`
(la clase que alimenta a `task2_baseline.py`, `task2_common.py`,
`task2_dinov2.py`, y este `task2_dpcn.py`) usa la convención correcta
codificada directamente en `fido/data/common.py`, no el argumento por
defecto de este script de diagnóstico standalone. No es, por sí solo, la
explicación de por qué las corridas previas (T2-R2/R3) dieron AUC=0.00.
