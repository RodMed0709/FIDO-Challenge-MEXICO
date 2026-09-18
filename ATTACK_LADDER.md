# Escalera de ataque — FIDO Challenge

Una hipótesis por peldaño. Cada peldaño se pre-registra (qué esperamos y por
qué) antes de correrlo, y se cierra con el resultado real — incluso si falló.
Un peldaño sin resultado registrado cuenta como no ejecutado.

**Estado**: `PLANEADO` · `CORRIENDO` · `LOGRADO` · `FALLÓ` · `DESCARTADO`

Esta escalera está informada por 3 revisiones de literatura completas
(`literature/01_*.md`, `02_*.md`, `03_*.md`, 2026-08-15) y por un hallazgo
crítico verificado: **arXiv 2603.25555** (Rohrmoser, Ghazaei, Sommersperger,
Navab — tres de los siete organizadores del challenge) resuelve exactamente
Task 1 sobre lo que casi con certeza es el mismo simulador. Sus cifras son
nuestro punto de referencia, no solo el leaderboard.

---

## Contexto del leaderboard (fase Competition, 2026-08-14)

| Task 2 — registración | score | | Task 1 — keypoints | score |
|---|---|---|---|---|
| cnielsen | 0.475 | | kwonmj | 0.619 |
| Alaa Senjab | 0.296 | | Alaa Senjab | 0.618 |
| oli4more | 0.280 | | akkkkk | 0.604 |
| kwonmj | 0.279 | | cnielsen | 0.588 |
| dancies | 0.112 | | dancies | 0.565 |
| akkkkk | 0.111 | | campana | 0.549 |
| alexlin | 0.047 | | Alex | 0.479 |
| tianmaxingkong, fatemeazami, campana, Alex | 0.000 | | tianmaxingkong | 0.418 |

**Referencia externa (Rohrmoser et al., Task 1)**: kp_dist 7.93±0.47 px
(multimodal), dMAE 128.32±20.03 µm (33.05 µm en régimen <1mm). No es un score
de AUC, pero da la vara de precisión alcanzable en este dominio exacto.

**Objetivos**: podio en Task 2 (meta 0.50+), competitivo en Task 1 explotando
la componente de distancia infra-explotada por el resto del campo.

---

## Refinamiento tras revisión adversarial (2026-08-15)

Tres agentes atacaron la escalera original desde ángulos distintos —
viabilidad/secuencia, fidelidad a la literatura, y modos de fallo concretos.
Informes completos en `literature/adversarial_0{1,2,3}_*.md`. Correcciones
aplicadas abajo; los tres informes coinciden en más de lo que discrepan.

**Hallazgo estructural más importante (viabilidad)**: fuera de `analysis/`,
`eval/` e `infra/`, **no existía ni una línea de código de modelo**. T2-R2 y
T1-R3 escondían medio día de arnés (dataset loader, geometría, loss,
GroupKFold) disfrazado de "bloqueado por pod". Ya se resolvió parcialmente:
`src/fido/geometry.py` (verificado byte a byte contra el scoring oficial,
`src/fido/tests/test_geometry.py`) y `src/fido/heatmap_decode.py` (DARK, UDP,
soft-argmax local) ya están escritos y no dependen de arquitectura ganadora.

**Corrección de un malentendido (modos de fallo)**: el tercer agente marcó como
contradicción que `RULES_OF_ENGAGEMENT.md` dice "distancia en píxeles reales
de B-scan" mientras `literature/02_*.md` infiere µm. **No es una discrepancia
entre mis documentos** — `RULES_OF_ENGAGEMENT.md` cita textualmente el
comentario del código de scoring oficial vendorizado ("true tool-tip-to-retina
pixel distance on the B-scan image"); el agente de literatura infirió µm sin
cruzarlo contra el código real. Se resuelve con datos, no con más
razonamiento: es exactamente lo que mide T1-R2 (el factor `a` del ajuste lineal
dice si la relación es 1:1 en píxeles o hay un factor de conversión).

**Cambios aplicados a la escalera** (detalle completo en los 3 informes):

1. **Insertados peldaños de arnés explícitos** — T2-R1.5 y T1-R2.5 — en vez de
   esconder la construcción de dataset/loss/split dentro de "bloqueado por pod".
2. **Reordenado T2**: puente por vasos (antes T2-R4) va ANTES que DPCN++
   log-polar (antes T2-R3) — menor riesgo de implementación, y el tercer
   agente señala que **la máscara de vasos del lado OCT puede no existir**
   (solo confirmada en `Stereo Left/.../Segmentation/arteriesorveins.png`, del
   lado fundus) — se verifica ANTES de construir el puente, no se asume.
3. **Añadido peldaño explícito para `oct_volume=None` en Task 2** — nadie lo
   tenía, y es el mismo patrón que dejó a 4/11 equipos en 0.000 en el
   leaderboard actual.
4. **Promovido template matching clásico** (RetinaMatch >94%, Noyel 96%,
   ambos sobre estructura vascular) a peldaño explícito, no descartado — es
   casi gratis una vez existe la segmentación de vasos y tiene mejor
   precedente empírico que varios métodos de deep learning que estaban más
   arriba en la prioridad.
5. **Recortes para los 5 días de fase Competition** (mover a backlog de Final
   Round, 20 días): backbone equivariante a rotación (T2-R6 — riesgo real:
   librerías `escnn`/`e2cnn` poco mantenidas contra PyTorch 2.7, no falta de
   kernels CUDA), HRNet desde cero (T1-R4), ensamble final (T1-R8), y
   transferencia secuencial entre tareas (X-R1).
6. **Marcado explícito de incertidumbre `[abs]`**: las cifras de rendimiento de
   DPCN++, IHN, CoMIR, VDD-Reg vienen de abstracts de arXiv sin PDF leído. Se
   tratan como orden de magnitud, no como benchmark verificado.
7. **T2-R1 no es un riesgo pendiente** — la estructura de 4 DOF ya se verificó
   sobre los 1214 casos reales de entrenamiento (no solo 5 del Mock Test),
   con `analysis/verify_task2_structure.py`. Un adversarial lo marcó como
   riesgo por leer solo la descripción corta del peldaño; queda aclarado aquí.
8. **Pendiente de resolver antes de cualquier submission real**: probar el
   contenedor contra el timeout global real (600s/100 casos) — ningún peldaño
   lo hacía explícito, y el reglamento da un solo tiro por task en la ronda
   final.

---

## Peldaños completados (fundacionales)

### R0 — Infraestructura y contrato de entrega
**Estado**: LOGRADO. Dataset íntegro en el volumen (83 GB, verificado por
checksum). Harness local (`eval/run_local.py`) probado con ambas tasks contra
el Mock Test. Submission de humo (`experiments/00-smoke-contract/`) confirma
que el contrato funciona: firma de 4 argumentos, `model_0.pth`/`model_1.pth`,
manejo de `oct_volume=None`. Detalle en `experiments/00-smoke-contract/README.md`.

### R2 — ¿El crosshair de Task 2 es detectable visualmente?
**Estado**: LOGRADO — **NO**. No está renderizado en la imagen. Descartado
como atajo.

### R3 — Estructura de la matriz de Task 2
**Estado**: LOGRADO. Similitud reflejada de 4 DOF verificada sobre 1214
snapshots: determinante negativo 100%, ortogonalidad 100% (±10°). Escala
160±13.7 px (NO constante — corrección de una medición previa). Forzar la
forma de 4 DOF dejaría un techo de AUC en 0.794 si se hiciera de forma pura.
Detalle completo en la versión anterior de este archivo (git history) y en
`docs/superpowers/specs/2026-08-14-fido-task2-design.md`.

### R4 — Literatura
**Estado**: LOGRADO. 3 revisiones completas, ~150 papers indexados entre las
tres. Hallazgo mayor: paper de los organizadores (2603.25555) verificado real.
Síntesis completa en memoria persistente (`fido-literature-synthesis`) y en
`literature/`.

---

# TASK 2 — registración iOCT→fundus (prioridad: campo más débil)

### T2-R1 — Confirmar la hipótesis de reflexión fija
**Estado**: PLANEADO
**Hipótesis**: el determinante es negativo en el 100% de los 1214 casos de
entrenamiento (ya verificado). Antes de construir cualquier método de
correlación de fase (que asume similitud propia), confirmar que aplicar un
flip fijo antes del solucionador reduce el problema a 4 DOF sin reflexión.
**Cómo**: trivial, ya verificado en `analysis/verify_task2_structure.py`. Este
peldaño es formalidad — se marca LOGRADO en cuanto se re-confirme sobre datos
frescos del volumen completo (no solo las anotaciones ya extraídas).
**Costo**: $0, CPU local.

### T2-R1.5 — Arnés de entrenamiento
**Estado**: LOGRADO
**Qué es**: no es una hipótesis científica, es infraestructura. Dataset loader
(fundus + proyección en-face desde los 128 slices), split GroupKFold por
escenario (10 grupos), empaquetado de submission, y verificación de que
`geometry.py` reproduce el scoring oficial byte a byte.
**Hecho**: `src/fido/geometry.py` + `src/fido/tests/test_geometry.py`
(compose_similarity, decompose_similarity, project_corners, corner_error,
corner_auc, ajuste cerrado de similitud — verificados contra el módulo de
scoring vendorizado y contra 5 GT reales del Mock Test, no reimplementados a
ojo). `src/fido/heatmap_decode.py` (soft-argmax local, DARK, generación de
heatmap GT no cuantizado). `src/fido/data/common.py` + `src/fido/data/task2.py`
(`Task2Dataset`: fundus, máscara de vasos del fundus, en-face + en-face de
densidad de vasos, matriz GT + parámetros decompuestos) — probado contra 5
casos reales del Mock Test en el pod, `__getitem__` da los shapes/dtypes
esperados.
**Bug real encontrado y corregido en `geometry.py`**: `compose_similarity(reflect=False)`
nunca invertía el signo de `b`, así que la rama "similitud propia" (sin
reflexión) no era una matriz ortogonal válida — `fit_closed_form_similarity`
no podía recuperarla. Nunca se había detectado porque el test que lo cubre
(`test_closed_form_recovers_known_transform`) jamás llegaba a correr: abortaba
antes en un test previo por falta de datos en el pod. Corregido; los 5 checks
de `test_geometry.py` pasan ahora contra el pod real.
**Falta**: probar con GPU real entrenando (no solo `__getitem__`), manejo
explícito de `oct_volume=None` (ver T2-R1.6, N/A para Task 2 en entrenamiento).

### T2-R1.6 — Manejar `oct_volume=None` en Task 2
**Estado**: PLANEADO — antes de cualquier submission, aunque no esté
documentado como frecuente en Task 2 (a diferencia de Task 1, donde es 10%
determinista)
**Hipótesis**: `RULES_OF_ENGAGEMENT.md` confirma que `load_oct_volume`
devuelve `None` si el directorio del volumen no existe. Una excepción no
manejada aborta los 100 casos completos, no solo ese caso — el patrón que
probablemente explica varios `0.000` del leaderboard actual.
**Qué hacer**: fallback determinista (predicción solo desde fundus, sin la
rama de en-face) que nunca lance excepción. Probar explícitamente con un caso
sintético de `oct_volume=None` en el harness local antes de cualquier subida.

### T2-R2 — Baseline: heatmap de centro + regresión de (θ,s)
**Estado**: LOGRADO (mecánicamente) — **AUC=0.0000 sobre datos reales
completos**, no cruza el umbral de <10px necesario. Resultado real, no bug.

**Resultado de la primera corrida (25 épocas, fold 0/5, 966 train/248 val)**:
`val_corner_auc=0.0000`, `val_mean_error` 203.29px → **98.74px**.

**⚠️ Esa corrida estaba condenada de antemano por un bug, no por capacidad.**
La conclusión original ("es techo de capacidad/tiempo de entrenamiento, no de
diseño roto") era **incorrecta**. Una auditoría adversarial del pipeline
completo (2026-08-17) encontró tres defectos estructurales, cada uno capaz de
forzar AUC=0 por sí solo:

**BUG 1 — el heatmap predice el CENTRO y se asignaba a la ESQUINA.** El pico
de la correlación cruzada marca dónde se alinea el *centro* de la plantilla
en-face sobre el fundus. El código lo asignaba directo a `(tx,ty)`, que en el
GT es la esquina `uv=(0,0)` del cuadrado unitario. Están separados por
`A@(0.5,0.5)`, es decir `s·√2/2`. **Medido sobre los 1214 casos reales:
113.1 ± 9.9 px** (predicción teórica con `s`=159.6: 112.9 px). El error de
validación observado era 98.74px — del mismo orden. El modelo ya localizaba
bien; el error era casi todo esta confusión de convención.
*Verificación decisiva*: alimentando el decodificador con localización y
parámetros **perfectos** (tomados del GT, sin ninguna predicción):

| decodificador | error de esquina | AUC |
|---|---|---|
| viejo (pico → `tx,ty`) | 112.88 px | **0.0000** |
| nuevo (pico → centro → esquina) | 1.81 px | **0.7848** |

O sea: el código anterior **no podía puntuar por encima de 0 aunque
entrenara perfectamente**. Ironía documental: el propio comentario del
modelo decía "la posición del mapa de correlación YA ES el centro del
template" y dos líneas después lo asignaba a la esquina.

**BUG 2 — kernel de correlación de lado PAR: sesgo de media celda.** Con
`padding=(Ht//2, Wt//2)` y plantilla de lados pares, la celda de salida `o`
corresponde al centro `o - 0.5`, no a `o`. Son 0.5 celdas × 16 px = 8 px por
eje → **11.3 px de error de esquina, por encima del umbral de 10 px del AUC
por sí solo**. Verificado numéricamente incrustando una plantilla conocida:
sin corrección el pico sale a (+0.50, +0.50) celdas del centro real; con
`-0.5`, exactamente 0.00.

**BUG 3 — plantilla con relación de aspecto 4:1 contra una huella cuadrada.**
El volumen OCT cubre una región cuadrada de retina (el GT es una similitud,
ortogonalidad 100%) pero se muestrea anisótropo: 128 slices × 512 A-scans. La
plantilla salía de 8×32 celdas mientras la huella real es ~160×160 px = 10×10
celdas. La correlación comparaba geometrías incompatibles y el pico nunca
podía ser nítido. Corregido remuestreando el en-face a `SCALE_REF²` antes del
encoder.

**Defectos de optimización corregidos en el mismo pase** (todos con número
medido, ninguno cosmético):
- `scale = softplus(raw)+1` arrancaba en **1.69** contra un objetivo real de
  ~160, así que el término de escala del MSE valía ~45 = **97% de la pérdida
  total** y monopolizaba el presupuesto de `clip_grad_norm_(1.0)`, dejando a
  los encoders sin gradiente. Ahora `scale = 160·exp(raw)` (error
  multiplicativo) y el MSE se compara en log.
- `coord_l1` en píxeles crudos producía normas de gradiente de miles contra un
  clip de 1.0 — el clip actuaba como un learning rate fijo y minúsculo, y lo
  que parecía "convergencia lenta" era el optimizador congelado. Normalizado
  por el tamaño de imagen.
- El BCE del heatmap con reducción media dejaba la señal de "dónde está" en
  1.6% del mapa (69 de 4225 celdas), es decir en el ruido numérico. Ahora con
  `pos_weight` y peso 1.0 en vez de 0.1.
- El GT del heatmap se generaba en `(tx,ty)` (la esquina) — se le pedía al
  modelo apuntar a un punto que la correlación no puede marcar. Ahora en el
  centro, derivado de la misma matriz GT.

**Resultado tras los fixes** (mismo split, mismo fold): 110.97px (ép. 5) →
94.60px (ép. 10) → **89.52px (ép. 15)**, bajando de forma monótona, contra
203→220→107 de la corrida anterior. AUC sigue en 0.0000: quitado el techo
estructural, ahora se mide el error de aprendizaje real.

**Siguiente límite identificado, aún no atacado**: la rotación se predice
desde un vector de features **promediado globalmente** (`mean(dim=(2,3))`), y
el promedio destruye justo la información de orientación. El error de esquina
por rotación es `≈ s·Δθ`, así que bajar de 10 px exige `Δθ < 3.6°` — muy
difícil desde un vector mean-pooled. Es el siguiente peldaño, no un bug.
**Cicatriz operativa de esta corrida**: 3 relanzamientos por problemas de
rendimiento (I/O síncrono sin `num_workers`, `include_vessel_enface=True`
innecesario, validar cada época) antes de llegar a un run que terminara en
tiempo razonable — detalle completo en `RESULTS.md`.
**Historial de peldaños previos, conservado abajo**:
**Hecho**: `src/fido/models/task2_baseline.py` (`FundusEnfaceHeatmapModel`) +
`src/fido/train/train_task2_baseline.py`. Arquitectura: encoder CNN chico
para fundus y para en-face, correlación cruzada espacial real (estilo SiamFC,
truco de conv agrupada `groups=batch_size` para que cada muestra del batch
solo vea su propia plantilla) → heatmap → soft-argmax global para (tx,ty),
cabeza de regresión separada para (cosθ,sinθ,s). Loss = error de esquinas
real (`corner_error`+`huber_saturated`) + L1 sin saturar sobre (tx,ty) para
el arranque + BCE del heatmap + MSE de (cosθ,sinθ,s).
**4 bugs reales encontrados y corregidos vía smoke test de overfit (5 casos
del Mock Test, 1 solo escenario — válido para probar mecánica, NO para medir)**:
1. `huber_saturated` da gradiente exactamente cero por encima de 15px de
   error (por diseño) — sin una pérdida sin saturar para el arranque
   (errores de cientos de px al empezar), el modelo nunca recibe señal de
   esa pérdida.
2. `local_soft_argmax_2d` (ventana de 7px alrededor del argmax duro) no deja
   que el punto predicho "salte" a la ubicación correcta desde un modelo sin
   entrenar — solo sirve para refinar una vez ya está cerca. Cambiado a
   `soft_argmax_2d` (global) para el entrenamiento desde cero.
3. La fusión con FiLM (vector global del en-face modulando el mapa del
   fundus) se estancó en ~65px de error incluso memorizando 5 ejemplos con
   300 épocas — la cabeza de rotación/escala convergía perfecto, la de
   posición no, porque FiLM no da ninguna señal espacial. Reemplazada por
   correlación cruzada real.
4. La correlación cruda explota en escala desde la primera pasada (BCE del
   heatmap ~838 en época 1). Ni `clamp` ni `tanh` lo arreglan — ambos tienen
   gradiente exactamente cero una vez el valor ya es enorme (confirmado:
   con ambos el heatmap se congeló en un valor exacto y dejó de entrenar
   para siempre). La normalización tipo instance-norm (restar media, dividir
   por desviación estándar de cada mapa) sí funciona — nunca pierde
   gradiente, se adapta a cualquier escala.
**Resultado del smoke test tras los 4 fixes**: sin colapsos, converge de
forma ruidosa a ~165-200px de error de esquinas sobre los 5 casos (300
épocas, batch completo). Todavía lejos de los <15px necesarios para AUC>0,
pero es tuning de hiperparámetros a partir de acá, no un bug de fondo.
**Dataset completo ya extraído (10/10 escenarios por task)**. Primer intento
de entrenar sobre datos reales murió con `PermissionError` en el escaneo de
`find_task2_cases` — no era un permiso roto persistente (verificado con
`find` sobre los 10 escenarios tras el `chmod -R` previo: todo legible), sino
la misma flakiness transitoria de MooseFS que mató T1-R5 (ver ahí). Corregido:
`find_task2_cases` captura `OSError` por caso y salta con aviso en vez de
tronar/colgarse; `Task2Dataset.__getitem__` reintenta con el siguiente índice
si un caso falla tras agotar los reintentos de `common.py`. Relanzado, 25
épocas, `checkpoints/task2_real`. Resultado real: pendiente.
**Original abajo, conservado por historial**:
**Hipótesis**: correlación cruzada entre features del fundus y features de la
proyección en-face → heatmap de `(tx,ty)` (el parámetro que pesa el doble en
la métrica) + cabeza separada para `(cos θ, sin θ, s)`. Loss = error de
esquinas exacto tras reconstruir M en forma cerrada.
**Por qué primero**: es el diseño más simple que usa la estructura del
problema (localización de template, no homografía de solapamiento alto).
**Expectativa pre-registrada**: 0.20–0.35 de AUC. Sirve de piso real (no
trivial) contra el que medir todo lo demás.
**Control**: contra predictor constante (AUC 0.000, ya medido).

### T2-R3 — Verificar máscaras de vasos en el lado OCT antes de construir T2-R4
**Estado**: LOGRADO — **sí hay clase de vaso limpia, sin ambigüedad**
**Resultado**: sobre todo el Mock Test de Task 2, los valores presentes en
`iOCT Microscope/Volume/<frame>/Segmentation/*.png` son
`{0,1,2,3,7,8,12}`, y contra el enum oficial (`GenericLabels` en
`vendor/fido/Dataset Explorer/constants.py`) mapean a:

| valor | clase | uso |
|---|---|---|
| 0 | Background | — |
| 1 | Ilm | referencia de profundidad |
| 2 | Rpe | referencia de profundidad |
| **3** | **ArteriesOrVeins** | **el puente para T2-R4 — mismo significado semántico que `Stereo Left/.../Segmentation/arteriesorveins.png` del lado fundus** |
| 7 | Liquid | — |
| 8 | Forceps | instrumento visible en el volumen (señal extra no explotada aún) |
| 12 | ToolMirrorOCTArtifact | artefacto de espejo del instrumento, NO vaso — el adversarial temía que esto se mezclara con vasos; no es el caso, son clases separadas |

**Costo real**: $0, 5 minutos, contra el Mock Test local. La preocupación del
adversarial (máscara "combinada" e inseparable) no se sostuvo — es una máscara
multi-clase estándar, trivial de aislar con `mask == 3`.

### T2-R4 — Puente por segmentación de vasos + template matching clásico
**Estado**: Motor 1 (template matching clásico) **REFUTADO por evidencia
oráculo** — ver abajo. Motor 2 (registro aprendido) sigue PLANEADO, pero
depende de resolver primero el hallazgo sobre `enface_vessel_density`.

**Motor 1 — resultado real (2026-08-17)**: `src/fido/models/vessel_template_match.py`
(NCC vía `cv2.matchTemplate`, grid search escala/ángulo, recuperación de
parámetros con `fit_closed_form_similarity` ya verificado) probado contra el
Mock Test (5 casos, máscaras GT de vaso PERFECTAS como entrada — no un
segmentador ruidoso, para aislar el método). **No es un bug de código**:
verificado con un caso sintético autoconsistente (`corner_error=0.09px`,
`match_score=0.9965` — la cadena `_build_patch`→`matchTemplate`→
`fit_closed_form_similarity` es matemáticamente correcta) y con un oráculo
sobre los casos reales (usando los parámetros REALES del GT, sin ninguna
búsqueda, la correlación entre el patch de `enface_vessel` alineado
correctamente y la región real del fundus da NCC≈0, ruido puro, en los 4
casos válidos). **La señal misma no correlaciona con el target, ni siquiera
en la respuesta perfecta** — `enface_vessel_density` (fracción de
profundidad ocupada por `ArteriesOrVeins` por columna, ~4-6% píxeles
no-cero) es demasiado dispersa y no reproduce la topología continua del
árbol vascular visible en el fundus. `corner_error` real: media=573.69px
(peor que T2-R2 sin entrenar, ~165-200px), `corner_auc=0.0000`.
**Hallazgo secundario, no bloqueante**: la escala real del GT en el Mock
Test (207-224 vía `decompose_similarity`) cae fuera del prior citado
(160±13.7px) — puede ser específico de `Scenario_12` (único escenario del
Mock Test) o indicar que el prior necesita reverificarse contra más
escenarios; no se investigó a fondo porque no cambia la conclusión
principal. Detalle completo, script del oráculo y del caso sintético en
`analysis/verify_vessel_template_match_synthetic.py`.
**Próxima hipótesis, no la investigada aquí**: revisar la construcción de
`enface_vessel_density` en sí antes de invertir en Motor 2 — ¿binarizar en
vez de promediar densidad? ¿dilatar/engrosar el patrón disperso? ¿usar otra
proyección del volumen (p.ej. max en vez de mean sobre profundidad)?

**Seguimiento (2026-08-17) — hipótesis de la señal en-face también
REFUTADA**: se probaron todas las variantes candidatas de la señal con el
MISMO oráculo (parámetros REALES del GT, sin búsqueda), extendido en
`analysis/verify_vessel_signal_candidates.py`:

| candidata | media oracle_NCC (n=4) |
|---|---|
| A. `enface_vessel_density` (fracción, actual) | 0.0071 |
| B. binarizada `(seg==3).any(axis=1)` | 0.0071 (idéntica a A — para máscara binaria por vóxel, fracción y presencia/ausencia correlacionan igual con cualquier target) |
| C. MIP (max) sobre densidad binaria | no se corrió: matemáticamente idéntica a B (`max(0/1)==any`) |
| D. binaria + dilatación k=2..8px (un solo lado) | 0.0122 → 0.0181, sin tendencia clara, todas <0.02 |

Ninguna cruza ni de lejos el umbral orientativo de 0.3. Dos verificaciones
adicionales antes de cerrar la hipótesis, para descartar "casi alineado pero
desplazado unos píxeles" en vez de "sin señal real":

- **Dilatación SIMÉTRICA** (patch OCT y máscara GT del fundus, no solo un
  lado) con kernel 0 a 21px, sobre un warp directo de la matriz GT completa
  (verificación independiente de `_build_patch`/bbox — de paso se encontró y
  corrigió un bug real en el oráculo original: usaba `(tx,ty)` como esquina
  superior-izquierda del patch cuando en realidad es la posición de la
  esquina `uv=(0,0)` del cuadrado unitario, que coincide con el bbox solo en
  rotación 0/90/180/270°; la corrección no cambia la conclusión — NCC sigue
  en [0.001, 0.010] hasta con 21px de tolerancia por lado).
- **Búsqueda local con traslación libre** (±30px escala, ±15° ángulo
  alrededor del GT, dilatación k=5 en ambos lados, pero dejando que
  `matchTemplate` busque la mejor posición en TODO el fundus): sube a
  NCC≈0.24–0.31, pero la posición ganadora **no coincide con la posición
  real del GT** en ninguno de los 4 casos (p.ej. caso `00066`: GT en
  `tx=328,ty=57`, mejor match encontrado en `(568,712)`). Confirma que el
  score alto es un falso positivo por autosimilitud del árbol vascular
  (curvas finas se parecen entre sí en cualquier parte de la imagen) —
  exactamente el modo de fallo detrás de `corner_error=573px` — y no una
  señal real desplazada unos píxeles.

**Conclusión**: la proyección en-face de la clase `ArteriesOrVeins` del
volumen OCT, en la posición/escala/ángulo EXACTOS del GT, no reproduce con
fidelidad el árbol vascular visible en el fundus — ni promediando, ni
binarizando, ni dilatando uno o ambos lados. No es un problema de
representación de la señal resoluble con transformaciones morfológicas
simples. **Motor 1 (template matching clásico sobre vasos) queda cerrado**
sin más iteración en esta forma. No se corrió `eval_vessel_template_match.py`
con ninguna señal nueva porque ninguna candidata superó el oráculo — según
el criterio pre-registrado, solo se justifica invertir en el pipeline
completo si el oráculo ya es razonable. Recomendación siguiente: escalar a
Motor 2 (registro aprendido, T2-R4 punto 2) o directamente a T2-R5/T2-R6, y
si Motor 2 también depende de la misma señal `enface_vessel`/`enface`
cruda, considerar combinarla con la silueta del instrumento (clase
`Forceps`/artefacto de espejo, ya visible en el en-face — señal no
explotada, mencionada en `RESULTS.md`) en vez de vasos solos. Script:
`analysis/verify_vessel_signal_candidates.py`.
**Original abajo, conservado por historial**:
**Hipótesis**: entrenar un segmentador de vasos en el fundus (las máscaras
vienen en el dataset) y registrar contra la proyección en-face (segmentada si
T2-R3 lo confirma, cruda si no) reduce la brecha de apariencia RGB↔gris.
Literatura: CoMIR bate a traducción GAN; VDD-Reg funciona con solo 3 máscaras
anotadas.
**Dos motores a probar, del más simple al más complejo** (fusión de la
escalera original con la promoción del tercer adversarial):
1. **Template matching clásico** (RetinaMatch >94% éxito, Noyel 96% de 271
   pares, ambos sobre estructura vascular) — casi gratis una vez existe la
   segmentación, mejor precedente empírico medido que varios métodos de deep
   learning. Correrlo primero como piso.
2. **Registro aprendido** sobre las probabilidades de vaso (no binarizadas).
**Expectativa pre-registrada**: +0.05 a +0.15 sobre T2-R2. Marcado `[abs]` en
la literatura fuente — tratar como orden de magnitud.
**Explícitamente descartado por la literatura**: traducir OCT→fundus con GAN
antes de matchear. No se intenta.
**Control**: contra T2-R2, mismo split.

### T2-R5 — DPCN++ adaptado (correlación de fase log-polar)
**Estado**: PLANEADO — backlog si el tiempo aprieta, T2-R4 tiene menor riesgo
**Hipótesis**: la correlación de fase diferenciable resuelve θ por búsqueda
global en el dominio log-polar (sin discontinuidad angular que aprender) y
desacopla θ/s/t. Debería ser más robusto a fallos catastróficos que T2-R2/R4,
que es lo que más cuesta en la métrica AUC(0..10) — un fallo de 300 px pierde
los 11 umbrales igual que uno de 50 px.
**Expectativa**: similar o mejor AUC que T2-R4, con menos varianza. Cifra de
"32.7 fps" de IHN está marcada `[abs]` en la fuente — no tratarla como medida
verificada.
**Riesgo documentado**: la correlación de fase estándar asume solapamiento
razonable en el espectro; con 2.4% de área común puede no aplicar directo
sobre la imagen completa — mitigación: aplicar en cascada tras T2-R4.
**Control**: contra T2-R4.

### T2-R6 — Cascada: localización gruesa → refinamiento iterativo
**Estado**: PLANEADO
**Hipótesis**: la métrica AUC(0..10) premia dos cosas incompatibles con un
modelo monolítico — robustez (nunca fallar catastróficamente) y precisión
sub-píxel. Cascada de dos etapas: búsqueda global robusta (el mejor de
T2-R4/T2-R5) + refinador iterativo de pesos atados estilo IHN sobre el
recorte, supervisado con L1/Huber sobre las 4 esquinas.
**Expectativa**: reduce la cola de fallos grandes sin sacrificar precisión
fina. Esperado: +0.05 a +0.10 sobre el peldaño anterior ganador. Cifra de
latencia de IHN marcada `[abs]` en la fuente — verificar en el contenedor
real, no asumir.

### T2-R7 — Backbone equivariante a rotación
**Estado**: BACKLOG (Final Round) — recorte del adversarial de viabilidad.
Riesgo real señalado por el adversarial de modos de fallo: no son kernels CUDA
faltantes, son librerías de investigación (`escnn`/`e2cnn`) poco mantenidas
que pueden no ser compatibles con PyTorch 2.7 — depurar eso puede tomar un día
completo. Se reintenta con los 20 días de Final Round si el presupuesto de
tiempo lo permite.
**Hipótesis** (sin cambios): con θ~U(0,2π) y solo 10 ojos de entrenamiento,
una CNN normal tiene que aprender la invariancia a rotación de los datos, que
no sobran. Sustituir el backbone por uno steerable (SE2-LoFTR) la da por
construcción.

### T2-R8 — Augmentación sintética por composición de transformaciones
**Estado**: PLANEADO — corre en paralelo a los anteriores, no es un peldaño
secuencial sino un ingrediente que se activa en cuanto se fija la arquitectura
ganadora de T2-R4/R5/R6.
**Hipótesis**: componer transformaciones 4-DOF muestreadas de la distribución
empírica (`s~N(160,13.7²)`, `t~N(470,130²)`, `θ~U(0,2π)`) sobre los 1214 pares
reales multiplica los datos efectivos sin inventar anatomía nueva.
**Cuidado documentado**: SIDAR advierte que homografías sintéticas puras
pueden ser "distorsiones triviales". Mitigación: la transformación real *es*
una similitud exacta, así que el riesgo geométrico es bajo; aumentar
apariencia (ruido, iluminación) agresivamente además de geometría.

### T2-R9 — Ensamble + TTA con rechazo por incertidumbre
**Estado**: PLANEADO — último peldaño antes de congelar
**Hipótesis**: consenso de homografías sobre recortes (CropTTA) detecta y
permite rescatar los fallos catastróficos, que es donde se pierde AUC de
golpe. Ensamble de 3-5 modelos de los peldaños anteriores.
**Restricción dura**: 600 s / 100 casos en Final Round. El I/O de cargar el
volumen se paga UNA VEZ por caso, no por cada pasada del ensamble/TTA
(corrección de un adversarial que temía lo contrario) — pero medir latencia
real en el contenedor antes de comprometerse al tamaño del ensamble sigue
siendo obligatorio, sobre todo si T2-R7 (backbone steerable) entra en juego.

### T2-R11 — Oráculo de señal en-face
**Estado**: LOGRADO — **PARCIALMENTE INVALIDADO por T2-R11b**

**Hipótesis**: si la convención en-face es correcta, los landmarks de vasos e
instrumento proyectados por la matriz GT deben coincidir con sus huellas en el
volumen más que un null uniforme.

**Resultado** (`analysis/measure_task2_oracle_signal.py`,
`experiments/70-t2-oracle/ORACLE_FULL.md`): 1214 casos, 0 saltados. El
self-test geométrico del crosshair dio error máximo **0.0000 px**. En
vasculatura, los ratios observado/null fueron **0.7831** (r=0), **0.9126**
(r=2), **1.0311** (r=5), **1.0253** (r=10) y **1.0165** (r=20). Las 9
proyecciones dieron z-score de NCC entre **-0.5427** y **+0.5233**. En
instrumento, **279/1214** casos (0.2298) tuvieron el keypoint dentro de la
huella; los ratios observado/null fueron **0.8750 / 0.4545 / 0.3846 / 0.5479
/ 1.2110**.

**Por qué queda parcialmente invalidado**: T2-R11b demostró que esta corrida
usaba la convención en-face equivocada. Sus negativos no pueden cerrar la
hipótesis de instrumento; el negativo de vasculatura sí sobrevive al repetir
con la convención ganadora.

**Control geométrico adicional**: aunque `properties.json` declara
`BscanScanningPattern: "Cross"`, el volumen medido es un ráster de 128 B-scans
paralelos. La correlación entre slices consecutivos tiene media **0.671** y
mínimo **0.617**, sin discontinuidad en el índice 64; la correlación entre la
media de slices 0-63 y 64-127 es **0.857**. La proyección en-face es
geométricamente válida.

### T2-R11b — Verificar la convención en-face
**Estado**: LOGRADO — **HALLAZGO: la convención anterior estaba mal**

**Resultado** (`analysis/verify_task2_enface_convention.py`,
`experiments/70-t2-oracle/CONVENTION.md`): 1214 casos; radios 0.005, 0.01,
0.02 y 0.04 como fracción del lado del cuadrado unitario. En el test de
instrumento (n=469 puntos, `null_n=93800`):

| Convención | 0.005 | 0.01 | 0.02 | 0.04 |
|---|---:|---:|---:|---:|
| `identity__keep_u__keep_v` (la usada) | 1.1747 | 1.1561 | 0.7491 | 0.6995 |
| `transpose__flip_u__flip_v` | 3.6364 | 3.4309 | 4.1775 | 3.1568 |

En crudo a r=0.02: observado **0.0682** contra null **0.0163**. El veredicto
automático `not_convention` es un fallo de la condición, no de los datos: el
script exigía un ganador único >=1.5 y dos variantes superaron el umbral.

**Matiz obligatorio**: el test de vasculatura no sube con la variante
ganadora; sus ratios son **0.9444 / 0.9055 / 0.9503 / 0.9895**. Corregir la
convención rescata la señal del instrumento, no la de vasos.

### T2-80 — Ablation por componentes
**Estado**: LOGRADO — fold 0, seed 0, 248 casos de validación

Checkpoint `task2_fixed/model_1.pth`, GroupKFold(5) fold 0:

| Variante | Error medio px | Mediana px | AUC |
|---|---:|---:|---:|
| modelo tal cual | 66.25 | 41.45 | 0.0048 |
| posición GT | 22.80 | 19.67 | 0.0480 |
| rotación GT | 57.95 | 35.25 | 0.0114 |
| escala GT | 62.68 | 37.12 | 0.0187 |
| rotación + escala GT | 53.89 | 30.99 | 0.0337 |
| posición + escala GT | 18.74 | 13.63 | 0.1459 |
| posición + rotación GT | 10.78 | 9.17 | 0.2416 |
| TODO GT (control) | 3.48 | 2.77 | 0.6389 |

Al sustituir cada componente, el error cae **43.45 px (65.6%)** por posición,
**8.31 px (12.5%)** por rotación y **3.57 px (5.4%)** por escala. Los errores
medianos son angular **5.7°**, escala **7.0%** y posición **32.9 px**.

**Consecuencia**: el techo del enfoque actual es **0.6389**, por encima del
líder del leaderboard (**0.475**). La posición explica el 66% del error. La
escala, que una hipótesis previa señalaba como cuello, es el componente que
menos pesa dentro de este fold.

Evidencia: `experiments/80-t2-diagnostics/component_metrics.json` y
`component_cases.jsonl`. El checkpoint usa `SCALE_REF=160` transductivo; este
cierre es diagnóstico, no evidencia OOF limpia ni autorización de submission.

### T2-80b — Control OCT-shuffle
**Estado**: LOGRADO — **GATE SUPERADO**

Mismo checkpoint, fold y seed que T2-80:

| Métrica | paired | OCT-shuffle | delta |
|---|---:|---:|---:|
| M1 top-1% | 0.003402 | 0.002457 | 0.000945 |
| M2 <=10 px | 0.068548 | 0.004032 | 0.064516 |
| M2 mediana px | 37.121 | 107.296 | -70.175 |

**El modelo sí usa contenido OCT específico del caso.** El gate
pre-registrado en `experiments/80-t2-diagnostics/PRE_REGISTRATION.md` exigía
que el shuffle degradara M1 y M2 para autorizar construir solver; ambos se
degradan. M2 cae 17x en recall y empeora 2.9x en distancia mediana.

**Matiz obligatorio**: M1 apenas cambia (0.003402 → 0.002457). Los
descriptores densos por celda casi no llevan información específica del caso;
el mapa de correlación agregado sí. El modelo localiza la región, pero no
clava el punto. Esto supera el gate de dependencia del OCT para construir el
solver, pero no el gate posterior de representación común M1/M2 `>=50%`.

Evidencia: `experiments/80-t2-diagnostics/correspondence.md` y
`correspondence_cases.jsonl`.

### T2-81 — Augmentación geométrica exacta de escala
**Estado**: LOGRADO (DONE / GO) — fold 0, seed 0

La única intervención fue `--scale-augment`, con `M_aug = H @ M`. El rango
`[0.819208, 1.220691]` se derivó exclusivamente de los 966 índices de train
(`log(scale)` q05 `4.851293`, q95 `5.170360`); los 248 índices de validación
son disjuntos y `selection_uses_mock=false`.

En el holdout sintético de dos extremos por caso (496 evaluaciones), el modelo
augmentado mejora AUC `0.000550 -> 0.006048`, error medio
`85.765 -> 65.305px`, mediana `61.056 -> 42.100px` y scale MAE
`31.163 -> 14.278px`: reducción relativa **54.184%**, frente al gate de 20%.
Por escenario, AUC sube en ambos: Scenario_09 `0.000745 -> 0.007700` y
Scenario_10 `0 -> 0.001399`; la caída máxima registrada es `-0.001399`, es
decir, ninguna caída. `gate_pass=true`: **GO**.

El entrenamiento corrió 50 épocas sobre 248 casos de validación por época.
El checkpoint `checkpoints/task2_scale_aug/model_1.pth` se seleccionó en la
época **38**, con `val_corner_auc=0.0092` y `val_mean_error=77.60px`; ninguna
época posterior superó ese AUC. La ruta del checkpoint consta en el log, pero
el binario no está entre los artefactos locales y por tanto no se inventa hash.

Evidencia: `experiments/81-t2-scale-augmentation/holdout_results.json`,
`scale_augmentation.json` y `train_run.log`. El `holdout_run.log` local existe
vacío; los resultados numéricos autoritativos son el JSON estructurado.

### T2-82 — Descriptores comunes CNN
**Estado**: ❌ **RECHAZADO** (2026-08-19, 10 épocas completas, fold 0/5, seed 0)

El gate pre-registrado exigía M1/M2 `>=50%` y **colapso bajo OCT-shuffle**.
Falla los dos, y falla el segundo de la peor forma posible: no hay colapso
porque no hay dependencia que colapsar.

| época | train_loss | m1_top1pct | shuffle_drop | m2_median_px |
|---:|---:|---:|---:|---:|
| 1 | 1.85e-2 | 0.0618 | +0.088 | 178.49 |
| 5 | 1.34e-6 | 0.0745 | **-0.567** | 156.44 |
| 6 | 9.07e-7 | 0.0757 | +0.151 | 157.95 |
| 8 | 4.95e-7 | 0.0662 | +0.076 | 164.19 |
| 10 | 3.74e-7 | 0.0573 | +0.025 | 187.30 |

Tres lecturas, todas negativas:

1. **`train_loss` cae 5 órdenes de magnitud (1.8e-2 → 3.7e-7)**. Eso no es
   aprendizaje, es colapso del objetivo contrastivo a una solución trivial.
2. **`shuffle_drop_points` oscila entre -0.567 y +0.151, es decir ruido
   alrededor de cero.** Barajar el OCT entre casos no degrada la métrica: el
   modelo **no está usando el OCT en absoluto**. Este es el control de T2-80b,
   que el baseline de T2-R2 sí superaba — el descriptor común lo pierde.
3. **`m2_median_px` empeora monótonamente tras la época 5** (156 → 187 px)
   mientras `m1_top1pct` también cae (0.0757 → 0.0573). No hay un checkpoint
   bueno que rescatar; la trayectoria entera va en la dirección equivocada.

**Conclusión**: la ruta "espacio de descriptores común aprendido por apariencia"
queda cerrada. No es un problema de hiperparámetros ni de épocas — es que la
señal que el objetivo pide no existe en los datos (ver T2-83, sección
Vasculature, y la memoria del oráculo de apariencia negativo sobre 1214 casos).

Evidencia: `experiments/82-t2-common-cnn/train_run.log` (JSON por época, con
desglose `per_scenario` para Scenario_09 y Scenario_10).

### T2-83 — Oráculo train-only de convención en-face
**Estado**: ✅ **PASS con matiz importante** (2026-08-19)

Verificar sobre datos de **train** (966 casos, `Mock used=false`) cuál de las 8
combinaciones `{identity,transpose} × {keep_u,flip_u} × {keep_v,flip_v}` alinea
la proyección en-face con el fundus. `gate_pass=true`; variante esperada y
confirmada: **`transpose__flip_u__flip_v`**.

**Instrumento** (312 puntos, ratio observado/null por radio):

| variante | r=0.005 | r=0.01 | r=0.02 | r=0.04 |
|---|---:|---:|---:|---:|
| **transpose__flip_u__flip_v** | 3.44 | 3.40 | **3.60** | **3.01** |
| identity__flip_u__flip_v | **4.41** | 3.33 | 2.33 | 1.78 |
| identity__keep_u__keep_v | 2.03 | 1.31 | 0.87 | 0.72 |
| las otras 5 | 0.52–1.75 | 0.36–1.36 | 0.23–0.99 | 0.37–0.91 |

**El matiz**: la evidencia separa con fuerza el **par de flips**
(`flip_u__flip_v` gana a las otras tres combinaciones en ambas transposiciones),
pero separa **transpose de identity con margen mucho más fino de lo que sugiere
comparar contra `identity__keep_u__keep_v`**. A r=0.005 `identity__flip_u__flip_v`
es incluso mayor (4.41 vs 3.44). Lo que de verdad favorece a transpose es la
**estabilidad a través de los radios**: se mantiene en 3.0–3.6 en los cuatro,
mientras identity decae 4.41 → 1.78. Con n=312 puntos (≈24 aciertos a r=0.04),
la barra de error no es despreciable.

**Vasculatura** (906 puntos): **ninguna variante da señal**. Los 32 ratios caen
en [0.81, 1.04], centrados en 1.0. La vasculatura no correlaciona en **ninguna**
orientación.

**Por qué esto importa más que el `gate_pass`**: el gate lo aprueba el
instrumento, un objeto pequeño presente en una fracción de los casos. La
vasculatura — la señal densa y omnipresente que cualquier modelo de apariencia
aprendería — es plana. Esto **predice y explica el fracaso de T2-82**, y es
independiente del oráculo de apariencia ya negativo sobre 1214 casos.
La orientación en-face está bien; lo que falta es señal explotable.

Evidencia: `experiments/83-t2-enface-convention/oracle_train.{json,md}`.

---

### T2-96 … T2-101 — El giro de Task 2: de registración a estimación de pose
**Estado**: ✅ **CARACTERIZADO** — vía identificada y cuantificada, **no desplegable todavía**

Cadena de peldaños del 2026-08-19, tras descubrir que el leaderboard tenía
siete equipos puntuando (líder 0.475) mientras nosotros dábamos Task 2 por
cerrada. Resumen, cada uno con su informe en `experiments/`:

| peldaño | qué estableció |
|---|---|
| **T2-98** | Las anotaciones traen poses 3D. Sobre 1214 casos **solo varían dos cosas**: `iOCT.Rotation` y `Eyeball.Rotation`. El microscopio es fijo y nada se traslada. La matriz GT se deriva de ahí (R² 0.975–0.994) |
| **T2-98b** (geometría exacta) | La matriz es afín y **solo depende de `iOCT.Rotation`** — la del ojo es prescindible (una esfera es invariante a rotaciones sobre su centro). `theta = -roll + theta0`, residuo 0.78°. AUC LOSO **0.1512**. La cadena física completa (rayo→esfera→pinhole) **no convergió** |
| **T2-99** | La heterogeneidad entre escenarios (AUC 0.0 a 0.68) se aísla a **8 coeficientes de proyección** que no transfieren. Recalibrándolos por escenario: AUC **0.7204**. Refutadas: hipótesis del tilt y del parámetro oculto por escenario |
| **T2-100** | Esos coeficientes se estiman desde el fundus con **una sola feature** (`profile_edge_r50`, radio del viñeteado óptico): AUC **0.3201**. Más features generalizan peor (sobreajuste con 9 escenarios/fold). Control nulo: percentil 98.8 |
| **T2-96** | Señal real que sobrevive el barajado: (tx,ty) desde el OCT solo R²=0.73, `iOCT.Rotation` R²=0.48, `Eyeball.Rotation` desde fundus R²=0.22 |
| **T2-101** | ⛔ **El número honesto**: sustituyendo la rotación GT por la predicha, **AUC 0.0004**, error 122.8 px |

**El cuello, cuantificado**: la rotación predicha tiene **14.3° de error mediano**
y el modelo geométrico colapsa a cero con **2–5°**. Casi dos órdenes de
magnitud. Los números 0.1512 / 0.3201 / 0.7204 son **techos condicionados a
`iOCT.Rotation` del GT** y no deben citarse como score esperable.

**Lo que queda abierto, y es concreto**: Task 2 se reduce a *estimar
`iOCT.Rotation` con error < 5°*. Con eso, la geometría ya verificada entrega
0.32–0.72. Es regresión de pose 3D desde el volumen, con 1214 casos más los
61,691 frames de Task 1 que también anotan la transformación. Un Random Forest
sobre 33 features da 14.3°; una CNN 3D entrenada en condiciones es otra cosa.
**Trabajo de Final Round, no de hoy.**

**Vías cerradas por el camino** (con números, no por impresión): correspondencia
de vasos, descriptores comunes, correlación de fase, instrumento como landmark,
prior de pose constante, y el crosshair renderizado.

---

# TASK 1 — keypoint + distancia herramienta-tejido

### T1-R1 — Confirmar mismo simulador que Rohrmoser et al.
**Estado**: LOGRADO — **respuesta parcial: probablemente NO el mismo dataset**
**Resultado**: solo **2** clases de instrumento en los 61,691 frames
(`ENDOILLUMINATOR`, `CANNULA`), no 4 como reporta Rohrmoser (dos fórceps +
cánula + endoiluminador). Las dos aparecen en TODOS los frames — no es
variedad entre casos, es que ambas herramientas están siempre presentes en
escena a la vez.
**Hallazgo nuevo, más valioso que la pregunta original**: sobre 30 frames al
azar (10 escenarios), **CANNULA es siempre** la herramienta activa (100 %) —
`ENDOILLUMINATOR.Meta['ILM Distance']` es siempre el centinela `2147484000`
(≈ int32 max, "no aplica"), mientras `CANNULA.Meta['ILM Distance']` tiene un
valor real. Y **`Ground Truth/Task 1[2] = CANNULA.Meta['ILM Distance'] × 100`
exacto en 30/30 casos** — no es una relación buscada, es literal.
**Consecuencia para T1-R2**: el keypoint y la distancia de Task 1 son
específicamente de la **CANNULA**, no de "el instrumento" genérico. Si el
B-scan trae ambas herramientas visibles (cánula + endoiluminador), el test
geométrico de T1-R2 debe aislar la cánula, no cualquier píxel de
`InstrumentInOCT` — riesgo de contaminar la medición si el endoiluminador
también cae en esa clase de segmentación. Verificar al correr T1-R2 a escala.
**No resuelto**: si el patrón "CANNULA siempre activa" es universal en las
61,691 muestras o solo en la mayoría — 30 es una muestra, no el total.
**Hipótesis**: el dataset FIDO usa el mismo simulador que arXiv 2603.25555
(69,134 frames, 20 vídeos, 4 clases de instrumento vs nuestros 61,691 frames).
**Cómo verificar**: contar clases de instrumento distintas en las anotaciones
(`Surgical Tool/<TIPO>`) — el paper reporta 4 (dos fórceps, cánula,
endoiluminador). Si coincide, sube la confianza de que sus cifras (kp_dist
7.93px, dMAE 128.32µm) son el techo realista a perseguir, no solo una
referencia lejana.
**Costo**: $0, ya tenemos las anotaciones extraídas.

### T1-R2 — Test geométrico de la distancia (30 min, altísimo valor)
**Estado**: LOGRADO — **R² = 0.9916 sobre 92,012 mediciones (61,691 frames)**

```
distancia_GT = 0.7320 * pixel_gap + 2.6779
pixel_gap: [13, 410]   distancia_GT: [5.2, 308]
```

CANNULA activa en el **100 %** de los frames (61,691/61,691, no solo la
muestra de 30). El intercepto (2.68) es pequeño frente al rango — la relación
es casi lineal pura, con un factor de escala (0.732) que no es 1:1 exacto en
píxeles (consistente con que el B-scan tiene su propia escala física, no
necesariamente 1 px = 1 unidad de GT).

**Consecuencia directa para T1-R5**: la cabeza de distancia se implementa como
**segmentación de cánula + ILM en el B-scan y medición geométrica**, no como
regresión distribucional (DFL). Es un problema de segmentación bien
supervisado — mucho menos varianza esperada que regresar el escalar a ciegas.
Con R²=0.99 sobre datos limpios, el techo de esta componente ya no es 0.0445;
depende de cuánto error de segmentación quede tras entrenar un modelo (no de
las mediciones oráculo con máscara GT, que es lo que se midió aquí).

**Original abajo, conservado por historial:**
**Hipótesis**: la distancia herramienta-tejido se puede MEDIR segmentando la
punta del instrumento y la ILM en el B-scan y tomando la separación axial,
en vez de regresarla a ciegas. Nota: una hipótesis parecida se probó
manualmente sobre 5 frames del Mock Test en R8 (versión anterior de este
documento) y **no cuadró** con una medición ingenua punta→ILM más cercana.
Este peldaño la reintenta con el método correcto de la literatura: localizar
la punta en la columna A-scan correcta y ajustar `dist = a·píxeles + b` por
mínimos cuadrados sobre muchos casos, no verificar caso por caso a ojo.
**Procedimiento** (`analysis/verify_task1_distance_geometry.py`, por escribir):
1. Para cada frame con OCT: fila más baja de la máscara `InstrumentInOCT`,
   columna `c`. Fila de `Ilm` en esa columna (o mediana en ventana ±5 col).
2. `p = z_ILM − z_punta` en píxeles del B-scan.
3. Ajustar `distancia_GT = a·p + b` por mínimos cuadrados. Reportar R², a, b.
4. Repetir con el segundo B-scan ortogonal si R² sale bajo.
**Interpretación pre-registrada**:
- R² > 0.95, b≈0 → medir geométricamente, problema resuelto con mucha menos
  varianza que una regresión ciega.
- R² alto, b≠0 → hay un offset (posible "capa objetivo virtual" entre ILM y
  RPE, documentado en Arikan et al.) — aplicar el offset.
- R² bajo → la distancia no es axial simple; probar norma euclídea de ambos
  B-scans ortogonales antes de rendirse.
**Por qué importa tanto**: la mejor constante posible da AUC 0.0445 en esta
componente (0.3 del score). Rohrmoser (SOTA publicado) tampoco entra cómodo
bajo 10 µm — así que ganar aquí, aunque sea poco en términos absolutos, vale
relativamente mucho contra un campo que probablemente está igual de bajo.
**Nota sobre unidades (corrección de un adversarial)**: `RULES_OF_ENGAGEMENT.md`
cita textualmente el código de scoring oficial — el GT es "**true
tool-tip-to-retina pixel distance on the B-scan image**", no µm. La literatura
(`02_task1_keypoints.md`) infirió µm por el rango numérico, sin cruzarlo contra
el código real. **Este mismo test lo resuelve con datos**: el factor `a` del
ajuste dice si la relación es literal 1:1 en píxeles (a≈1) o si hay un factor
de conversión oculto. No asumir ninguna de las dos antes de correrlo.
**Estado de ejecución**: script escrito (`analysis/verify_task1_distance_geometry.py`),
probado sobre el Mock Test (insuficiente: solo 2 mediciones válidas de 3 casos
disponibles). Extrayendo máscaras de segmentación de B-scan del dataset
completo (`data/_bscan_seg/Task 1/`, en curso) para tener volumen suficiente.

### T1-R2.5 — Arnés de entrenamiento (análogo a T2-R1.5)
**Estado**: LOGRADO
**Qué es**: dataset loader (fundus + 2 B-scans), GroupKFold por escenario,
generación de heatmap GT no cuantizado, empaquetado de submission. Comparte
`src/fido/heatmap_decode.py` con Task 2.
**Hecho**: `src/fido/data/task1.py` (`Task1Dataset`: fundus, 2 B-scans +
segmentación remapeada a 3 clases, `has_oct` con fallback a ceros si falta la
carpeta Bscan, keypoint + distancia del GT) — probado contra 5 casos reales
del Mock Test. `src/fido/models/unet_bscan_seg.py` (UNet liviano
fondo/Ilm/InstrumentInOCT + `distance_from_segmentation`, réplica batched del
método ya verificado en T1-R2) — forward pass probado con GPU real en el pod,
shapes correctos.

### T1-R3 — Baseline: encoder CNN desde cero + heatmap sub-píxel (comparación A/B vs DINOv2)
**Estado**: LOGRADO — **CNN desde cero gana el A/B contra DINOv2**

**Resultado real**: la CNN desde cero alcanzó `val_keypoint_auc=0.8537`.
DINOv2 alcanzó su mejor `val_keypoint_auc=0.8005` en la época 6, con
`val_mean_error=1.72 px`, n=14399. DINOv2 pierde también en Task 2:
`val_corner_auc=0.0000` en las épocas 30, 35 y 40, mientras
`val_mean_error` subió 103.25 → 125.17 → 136.60 px. **Conclusión**:
DINOv2 pierde en las dos tasks y deja de ser el candidato principal.

**Arnés**: verificado con smoke test real, sin bugs de código encontrados
(0 bugs, a diferencia de T2-R2/T1-R5/T2-R1.5).
**Hecho**: `src/fido/models/task1_keypoint.py` (`Task1KeypointModel`:
`SimpleEncoder` — 5 `ConvBlock`s con downsample x16, `base_channels=32`,
`n_downsamples=4` — + cabeza `Conv2d` 1x1 de heatmap + `soft_argmax_2d`
GLOBAL) + `src/fido/train/train_task1_keypoint.py` (`compute_loss` = L1 sin
saturar sobre (x,y) + BCE del heatmap ponderado 0.1, `evaluate` con
`keypoint_auc` vía `corner_auc`). Es el backbone CNN-desde-cero del A/B contra
DINOv2 mencionado en `NOW.md`, no un descuido de arquitectura.
**Verificación del contrato de datos (antes de correr nada)**: cruzado
`Task1Dataset.__getitem__` contra `compute_loss`/`evaluate` — `batch['fundus']`
(3,1024,1024) float32 [0,1] y `batch['keypoint']` (2,) float32 coinciden
exactamente en keys/shapes/dtype. Confirmado con datos reales (no asumido):
`Ground Truth/Task 1[:2]` en el JSON es literalmente `Keypoints/Cannula
SRI/Tip` — orden **(x, y)** en píxeles del fundus 1024×1024 (verificado
también contra `RULES_OF_ENGAGEMENT.md`: `"keypoints": [x, y]` en el formato
de retorno de inferencia). El fundus real (`Stereo Left/<frame>/microscope.png`)
se comprobó con PIL: **1024×1024 RGB exacto**, así que el `fundus_size=1024`
hardcoded en `forward()`/`compute_loss` no es un valor asumido a ciegas, es
correcto.
**Smoke test de overfit** (5 casos reales del Mock Test, 2 escenarios —
insuficiente para GroupKFold real con `n_folds` por defecto, se usó la rama
de aviso automático "train=val completo"):
- Producción (`train_task1_keypoint.py`, mini-batch=2, 29 épocas antes de
  cortar por tiempo): sin crashes, sin NaN. `val_mean_error` ruidoso pero con
  piso descendente (163px → rango 60-90px hacia la época 27-28);
  `val_keypoint_auc` no nulo en algunas épocas (máx. 0.09).
- Diagnóstico adicional en memoria (mismo modelo/loss/seed, batch completo de
  5, para aislar de la sobrecarga de I/O y confirmar determinismo — dos
  corridas independientes coinciden bit a bit en el paso 40): `loss` cae de
  151.2 (paso 1, sin entrenar) a 31.3 (paso 60); al menos un ejemplo individual
  baja de >150px a **3.0px** de error; `keypoint_auc` llega a **0.145** (paso
  40). Esto descarta las categorías de bug del patrón conocido en este
  proyecto — no hay gradiente muerto, no hay colapso del heatmap, no hay
  pérdida saturada sin señal al arrancar.
**Ningún bug de código encontrado.** A diferencia de T2-R2 (4 bugs),
T1-R5 (2 bugs) y T2-R1.5, este módulo corrió limpio en su primer smoke test.
No se inventó ningún bug para justificar la revisión — se buscó activamente
(convención (x,y) vs (row,col), heatmap fuera de grilla en el borde, escala
de logits) y no apareció ninguno real.
**Hallazgo real, no un bug**: la convergencia sobre los 5 ejemplos es
**ruidosa/no monótona** incluso a decenas de pasos (p.ej. `mean_error` sube de
45px en el paso 60 a 74px en el paso 80 antes de volver a bajar). Root cause
confirmado con instrumentación (no adivinado): la norma del gradiente CRUDO
(antes de `clip_grad_norm_`) es enorme y **crece** durante el entrenamiento —
751 (paso 1) → 1891 → 4492 → 7469 → 5160 (paso 40) — 3-4 órdenes de magnitud
por encima del techo `max_norm=1.0` que se aplica en cada paso. Esto significa
que casi todos los pasos están dominados por el presupuesto fijo del clip, no
por la dirección/magnitud real del gradiente de la pérdida. Se descartó
explícitamente la hipótesis de "heatmap explotando en escala" (el bug ya visto
en T2-R2): `heatmap_logits.abs().max()` se mantiene acotado y crece
suavemente (8.4 → 12.3 en 40 pasos), y la norma de los pesos de la cabeza de
heatmap casi no se mueve (0.580 → 0.586) — el clip está haciendo exactamente
lo que su comentario dice que debe hacer (evitar el colapso visto en el
proyecto hermano), a costa de convergencia lenta/ruidosa en este régimen de
lote completo sobre 5 ejemplos.
**Riesgo abierto para el entrenamiento real en el pod**: si con miles de
ejemplos reales y mini-batches la norma cruda del gradiente sigue siendo así
de grande relativa a `max_norm=1.0`, la convergencia podría ser más lenta de
lo esperado dentro del presupuesto de épocas planeado. Recomendación:
vigilar la curva de `train_loss`/`val_keypoint_auc` temprano en la corrida
real; si se estanca, subir `--heatmap-weight`/`max_norm` (actualmente
hardcoded a 1.0 en el loop de entrenamiento, no expuesto como flag) es la
primera palanca a probar, no un rediseño de arquitectura.
**Hipótesis original (arnés ya escrito, arquitectura ya decidida)**: el
pipeline con decodificación sub-píxel (soft-argmax global) supera una
regresión directa ciega. La literatura en este dominio exacto (Laina et al.)
mide que regresión directa es la peor variante.
**Cascada de dos etapas** (justificada por la concentración de la distribución
medida: x∈[191,688], y∈[239,772]) — pendiente, no bloqueante para este
peldaño: pasada gruesa en 1024² completo → recorte 256² a resolución nativa
para la pasada fina.
**Expectativa pre-registrada**: keypoint_auc > 0.60 sobre datos reales
(calibrado contra que el líder actual, con `distance_auc` probablemente bajo,
ronda ~0.85-0.88 en esa componente para llegar a 0.619 total) — el smoke test
de 5 ejemplos no mide esto, solo confirma que el arnés no tiene bugs
estructurales.
**Control**: contra R00 (smoke, AUC 0).
**Cierre del A/B**: ya se entrenó sobre el dataset completo. La convergencia
real resuelve la pregunta abierta a favor de la CNN desde cero.

### T1-80 / T1-85 — Baselines limpias: CNN 20 épocas vs ResNet-18+FPN
**Estado**: ✅ **MEDIDO — hipótesis del backbone RECHAZADA** (2026-08-19)

Dos corridas de 20 épocas sobre el mismo split (n=14399 en validación),
lanzadas en paralelo en la misma GPU.

| corrida | mejor `val_keypoint_auc` | época del mejor | AUC final (ep20) | `val_mean_error` final |
|---|---:|---:|---:|---:|
| T1-80 CNN desde cero | **0.8539** | **5** | 0.8537 | 1.03 px |
| T1-85 ResNet-18 + FPN (BF16) | 0.8534 | **2** | 0.8512 | 1.11 px |

**Los dos resultados dicen lo mismo, y no es lo que la hipótesis esperaba:**

1. **El backbone no mueve la aguja.** ResNet-18+FPN pre-entrenado da 0.8534
   contra 0.8539 de la CNN desde cero: diferencia de 0.0005, ruido. Sumado al
   A/B previo (DINOv2 `0.8005`), son **tres arquitecturas distintas — CNN
   scratch, ResNet-18+FPN pre-entrenado, ViT auto-supervisado — y el techo se
   queda en ~0.853**. La capacidad del extractor de features no es el cuello.
2. **La saturación es inmediata.** El mejor checkpoint llega en la época 5 (CNN)
   y en la **época 2** (ResNet). Las 15-18 épocas restantes bajan `train_loss`
   (0.61 → 0.575 y 0.545 → 0.537) sin mover el AUC de validación: memorización
   pura. Todo lo entrenado más allá de la época ~5 es GPU quemada.
3. `heatmap_bce` en el ResNet es 0.0018, un orden de magnitud menor que en la
   CNN (0.0163), y **aun así no gana AUC**. El heatmap ya está prácticamente
   resuelto; el error residual no vive ahí.

**Consecuencia para la escalera**: T1-R4 (HRNet) y T1-R8 (ensamble de
backbones) pierden casi toda su justificación esperada — atacan el eje que
acaba de demostrarse plano tres veces. El techo de ~0.853 es del
**planteamiento** (formulación de la tarea, resolución, definición del target,
o el propio piso del AUC), no de la arquitectura. Antes de gastar otra corrida
en un backbone nuevo hay que medir **de qué está hecho ese 0.147 restante**,
igual que T2-80 hizo para Task 2.

Evidencia: `experiments/80-t1-clean-baselines/cnn_run.log`,
`experiments/85-t1-cnn-fpn/resnet_run.log`.

### T1-R4 — HRNet desde cero, en paralelo
**Estado**: BACKLOG (Final Round, no Competition) — recorte del adversarial de
viabilidad. ROI bajo para los 5 días que quedan de Competition frente a
asegurar T1-R2/R3/R5/R6 primero; se retoma con los 20 días de Final Round si
la CNN ganadora de T1-R3 no alcanza el nivel esperado por sí sola.
**Hipótesis**: HRNet mantiene resolución alta nativamente (sin el problema de
stride 14 de DINOv2) y ganó CATARACTS 2020, el challenge de dominio más
cercano al nuestro. Wood et al. demuestran que CNN desde cero + augmentación
fuerte puede batir a preentrenados en landmarks sobre renders sintéticos.
**Expectativa**: competitivo con T1-R3, posible ganador en la cola de casos
difíciles donde el stride de DINOv2 penaliza más.
**Control**: contra T1-R3, mismo split (GroupKFold por escenario).

### T1-R5 — Cabeza de distancia: geométrica (si T1-R2 lo confirma) o DFL
**Estado**: CORRIENDO — primer número real sobre dataset completo:
`val_distance_auc=0.5681` (época 2, parcial); relanzado tras fix de I/O
**Hecho**: `src/fido/train/train_task1_unet.py` — entrena `UNet` sobre los 2
B-scans (aplicado independiente a cada uno), mide accuracy de segmentación +
`distance_from_segmentation` + AUC de distancia con el ajuste ya verificado.
**2 bugs reales encontrados y corregidos vía smoke test (5 casos del Mock
Test, 2 escenarios reales — esta vez sí alcanzó para un GroupKFold real,
n_folds=2)**:
1. `train_loss=nan` a partir de cierta época, permanente: un batch con CERO
   píxeles válidos (todas las muestras con `has_oct=False`) hace que
   `CrossEntropyLoss(ignore_index=-1)` divida entre cero, y un solo paso con
   gradiente NaN corrompe los pesos para siempre (ni `clip_grad_norm_` salva
   un NaN). Corregido: saltar el batch entero si no tiene ningún píxel
   válido, antes de `backward()`.
2. **Desbalance de clases extremo, confirmado con números reales**: fondo
   99.5% de los píxeles, Ilm 0.46%, InstrumentInOCT (cánula) **0.008%**
   (132 píxeles de 1.57M). Con `CrossEntropyLoss` sin pesos, el modelo
   converge a predecir fondo en el 100% de los píxeles — incluso
   sobre-entrenando directo sobre las mismas 5 muestras. Corregido con pesos
   de clase por frecuencia inversa calculados de los datos reales de cada
   corrida (no un número inventado).
**Resultado tras los fixes**: sin crashes, el modelo ya predice Ilm/cánula en
algunos casos (antes 0/60 épocas con casos OCT válidos detectados, ahora la
mayoría de épocas sí). Error de distancia todavía enorme (~460-555px, lejos
de los <15px que dan AUC>0) — esperable con solo 5 ejemplos totales y 132
píxeles de cánula en todo el dataset de entrenamiento disponible ahora mismo.
**Primer resultado real (dataset completo, 15 épocas planeadas, fold 0/5)**:
`val_distance_auc=0.5681` en época 2 sobre 12,566 casos — ~12.8x el piso
trivial (0.0445). Corrida murió en época 3 por `OSError: [Errno 6] No such
device or address` en `Image.open` — I/O transitorio de MooseFS (fs de red
del pod), no bug de código; el mismo tipo de error mató T2-R2 con
`PermissionError` en otra parte del stack. **Root cause confirmado, no
adivinado**: reintentos (3 intentos, 0.3s) en `load_rgb`/`load_grayscale`/
`load_label_map` y en el listado de PNGs de volumen (`src/fido/data/common.py`,
decorador `_retry_on_io_errors`). Checkpoint de época 2 conservado
(`checkpoints/task1_real/model_0.pth`), relanzado desde cero con el fix (15
épocas, `infra/orchestrate_retry.sh`). Detalle completo en `RESULTS.md` R01.
**Original abajo, conservado por historial**:
**Si T1-R2 da R²>0.95**: implementar como módulo de segmentación + medición
geométrica, con una pequeña red de corrección residual entrenada end-to-end.
**Si no**: regresión distribucional (DFL sobre bins), ponderada hacia el
régimen cercano a 0 (los casos lejanos no son recuperables para el AUC).
**Expectativa**: distance_auc > 0.15 sería ya una mejora sustancial sobre el
piso de 0.0445, y valdría +0.03 al menos en el score final por cada 0.1 de
mejora.

### T1-R6 — Robustez al 10% sin OCT
**Estado**: PLANEADO
**Hipótesis**: modality dropout (p≈0.2-0.3, más alto que el 10% de test) +
embedding aprendido de ausencia (no ceros) + pérdida de alineación
contrastiva entre representación con/sin OCT.
**Explícitamente descartado**: entrenar dos modelos separados (duplica costo
de inferencia, la literatura de missing-modality lo encuentra inferior).
**Verificación obligatoria**: medir el `masked_dMAE` (error solo en casos sin
OCT) por separado — Rohrmoser mide que es 2.4-3× peor que el global incluso
con su mecanismo de robustez. Calibrar expectativas con ese número.

### T1-R7 — Cabezas auxiliares de segmentación concatenadas
**Estado**: PLANEADO
**Hipótesis**: concatenar los logits de segmentación (vasos, ILM, instrumento
— pre-softmax) a los features antes de la cabeza de heatmap, receta exacta de
Laina et al. en este dominio (>90% a umbral 20px con esta técnica).
**Control**: contra T1-R3/T1-R4 sin esta técnica, mismo split.

### T1-R8 — Ensamble DINOv2 + HRNet + TTA
**Estado**: BACKLOG (Final Round) — depende de T1-R4 (también en backlog).
Se activa solo si sobra tiempo tras asegurar T1-R2/R3/R5/R6/R7.

---

## Transferencia entre tareas

### X-R1 — Encoder de Task 1 como init de Task 2
**Estado**: BACKLOG (Final Round) — recorte del adversarial de viabilidad.
**Hipótesis**: usar el encoder ya entrenado sobre los 61,691 frames de
microscopio de Task 1 para inicializar la rama de microscopio de Task 2 es
transfer *dentro del dominio* (mismo tipo de imagen, misma cámara), que sí
funciona, a diferencia de multi-task simultáneo (que la literatura desaconseja
por interferencia de gradiente con datasets de tamaño 50:1 distinto).
**Nota**: esto es transferencia secuencial, no compartir pesos en producción.
Los dos modelos finales siguen siendo independientes (`model_0.pth`,
`model_1.pth`).

---

## Reglas transversales de esta escalera

1. **GroupKFold por escenario, siempre.** 10 escenarios confirmados en Task 2
   (10 ojos); el número exacto de escenarios distintos en Task 1 está
   pendiente de T1-R1 (probablemente 10 también, a confirmar, no asumir —
   corrección de una sobregeneralización señalada por un adversarial). Split
   aleatorio de frames o casos da un CV mentiroso — frames del mismo escenario
   comparten anatomía casi exacta.
2. **Ningún peldaño usa el Mock Test para seleccionar hiperparámetros**, solo
   para el score agregado final antes de una submission real
   (`CONSTITUTION.md` §3).
3. **Reservar ≥5 submissions para la Final Round.** De las 20 de Competition,
   máximo 15 se gastan en calibración.
4. **Validar el contenedor exacto** (`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`)
   antes de cada submission real — minimizar dependencias de terceros
   (flash-attn, xformers) que puedan no tener binarios `sm_120`.
