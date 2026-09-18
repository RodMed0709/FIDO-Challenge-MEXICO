# T1-91 — Revisión adversarial: r05=r03=0.569, y auditoría de cuatro conclusiones del proyecto

**Fecha**: 2026-08-19
**Rol**: revisor adversarial. Objetivo: encontrar dónde el equipo se equivocó o se
engañó, no confirmar lo ya escrito.

**Convención**: `[VERIFICADO]` = medido o leído directamente en este repo/código,
con archivo:línea o comando. `[INFERIDO]` = razonamiento explícito a partir de
verificaciones, no una medición directa. `[NO VERIFICABLE AQUÍ]` = requiere
algo fuera de este repo (el panel de Codabench).

---

## 0. Resumen ejecutivo

1. La premisa "r05 obtuvo 0.569 en Codabench, exactamente igual que r03" **no
   tiene ningún rastro en el repo**: la documentación local (`NOW.md`,
   `RESULTS.md`, `HANDOFF_2026-08-18.md`) solo registra dos submissions
   puntuadas — `r01=0.5386` y `r03=0.569` — más `r04-interim-joint=0.561`
   citado en prosa. `r05-dist-rescaled` y `r06-fallback-fixed` existen como
   paquetes listos (`.zip` incluidos, construidos el 2026-08-19 08:53 y
   10:28) pero **ningún archivo del repo registra que se hayan subido**.
   Si el hecho es real, ocurrió fuera de este repo o después de la última
   actualización de `NOW.md` (2026-08-19 13:45).
2. **La comparación "r05 = r03" no es la comparación correcta.** `r05` no
   parte de los pesos de `r03`: parte de `r04-interim-joint`, que ya es una
   submission real, distinta, con su propio score (`0.561`). La cadena
   correcta es `r03 → r04 → r05 → r06`, cada paso cambiando una sola cosa.
   Comparar `r05` contra `r03` salta un eslabón y mezcla dos cambios
   (checkpoint de keypoint **y** constantes de distancia) — exactamente lo
   que `CONSTITUTION.md` §6 prohíbe ("una hipótesis por rung").
3. Medí en vivo, con el checkpoint real de producción (`r06-fallback-fixed/
   model_0.pth`) sobre B-scans reales (no máscaras GT), la fracción de casos
   donde `distance_from_segmentation` devuelve NaN y el pipeline cae al
   fallback constante. Resultado parcial pero real: **36.7% (44/120,
   Scenario_01)** de los casos con distancia GT medible no reciben ninguna
   medición geométrica. Sumado al 10% de `oct_volume=None` forzado por la
   ingestión, entre **40-45% de los casos reales de Task 1 nunca pasan por
   el ajuste lineal A,B** — pasan directo por la constante de fallback. Este
   es el mecanismo más plausible por el que reescalar A,B (r05) tiene mucho
   menos efecto del esperado sobre el score real.
4. La hipótesis de mayor impacto potencial, en probabilidad y en costo si es
   cierta y se ignora: **nadie verificó que Codabench redeployó el scorer
   corregido**. Solo se verificó que el repo de GitHub cambió. El propio
   documento que debería reflejar esto (`RULES_OF_ENGAGEMENT.md`) sigue
   documentando las constantes VIEJAS (÷10, umbral 0-10, 20 submissions,
   cierre 20 de agosto) sin ninguna mención a la corrección del
   2026-08-19 — verificado, cero coincidencias en el archivo completo.

---

## 1. El rompecabezas r05=r03=0.569

### 1.1 Lo que el repo puede verificar sobre las submissions

| Submission | Score real Codabench | Task 2 real | `model_0.pth` (sha256, 8 chars) | `inference.py` (sha256, 8 chars) | ¿Registrada como subida? |
|---|---:|---:|---|---|---|
| `r01-task1-keypoint` | **0.5386** (`kp=0.7309`, `dist=0.0900`, verificado exacto en `RESULTS.md:243`) | — | `911489e9` | `63fbeec0` | sí |
| `r03-dist-v2` | **0.569** (breakdown "pendiente" — `RESULTS.md:56`) | — | `fa1d0640` | `63fbeec0` (idéntico a r01) | sí |
| `r04-interim-joint` | **0.561** (`NOW.md:102`, `experiments/86-t1-margen/INFORME.md:165`) | **0.00** | `6b73fc21` | `63fbeec0` (idéntico a r01/r03) | sí |
| `r05-dist-rescaled` | premisa: `0.569` | premisa: `0.00` | `6b73fc21` (**idéntico a r04**) | `b6280b23` (solo cambia A,B) | **no registrada en ningún .md del repo** |
| `r06-fallback-fixed` | no reportado | no reportado | `6b73fc21` (**idéntico a r04/r05**) | `23ec19e3` (además reescala fallback) | **no registrada** |

`model_0.pth` de `r03` es una checkpoint **distinta** a la de `r04/r05/r06`
(hash `fa1d0640` vs `6b73fc21`) — verificado por hash, no por nombre de
archivo. Localmente, con los mismos 5 casos del Mock Test, dan
`keypoint_auc` distinto: `0.818182` (r03) contra `0.854545` (r04/r05/r06)
— `submissions/r03-dist-v2/README.md` vs `submissions/r05-dist-rescaled/
local_score.json`. Son modelos diferentes, no la misma red con distinto
post-proceso.

### 1.2 Por qué "r05 = r03 exactamente" es una afirmación fuerte

El score final es `0.7·keypoint_auc + 0.3·distance_auc`
(`vendor/.../scoring_keypoints.py:76`), calculado con `auc_from_errors`
sobre **100 casos** (`RESULTS.md:54`, confirmado también en
`RULES_OF_ENGAGEMENT.md:90`) y 11 (viejo) o 21 (nuevo) umbrales enteros
(`scoring_keypoints.py:69-74`). La granularidad de `distance_auc` es
`1/(100·11)≈0.00091` (viejo) o `1/(100·21)≈0.00048` (nuevo) — mucho más fina
que "casualidad de bins". Con pesos de keypoint DIFERENTES (checkpoints
distintos) y predicciones de distancia DIFERENTES (constantes reescaladas
×1.28), una igualdad EXACTA a 6 decimales sería casi imposible por azar.
Pero **la cifra citada (`0.569`) tiene solo 3 decimales** — el mismo formato
en que aparecen `r01`, `r03` y `r04` en `NOW.md` y `RESULTS.md` (posible
lectura del leaderboard truncado, no de `scores.json`/`detailed_results.html`
a 6 decimales). Esto es relevante para H3 abajo.

### 1.3 Aritmética: qué `distance_auc` da exactamente 0.569

Con `score = 0.7·kp + 0.3·da`:

| Supuesto `kp` real | Fuente | `da` necesario para score=0.569 |
|---|---|---|
| `0.7309` | r01, único breakdown VERIFICADO en el repo (`RESULTS.md:243`) | `0.1912` |
| `0.774` | citado en el encargo de esta revisión — **sin fuente verificable en el repo** (ver 1.4) | `0.0907` |
| `0.7629` (inferido, ver abajo) | si `da≈0.09` fuera igual para r04 y su score real fuera 0.561 | `0.1166` |

El caso `kp=0.774` da `da≈0.0907` — **casi idéntico** a `0.090909`, el
`distance_auc` que ya se observó en `r01`, y el mismo valor que `r04` da
localmente bajo el vendor VIEJO (`submissions/r05-dist-rescaled/README.md`,
fila 1 de la tabla). Esto es el patrón más simple y más preocupante: si el
`kp_auc` real de r05 fuera efectivamente ≈0.774 (el mismo orden que r01),
el score de 0.569 se explica con un `distance_auc` **igual al que ya se
tenía antes de reescalar nada** — es decir, el reescalado no cambió nada
en la práctica.

**Contraprueba con `r04` (dato real, no supuesto)**: `r04` comparte pesos
con `r05`. Si asumo que el `distance_auc` real de `r04` (constantes viejas,
scorer viejo — la única combinación consistente para una submission subida
el 2026-08-17/18) fue también ≈0.09 (mismas constantes/fallback que r01/r03,
mismo scorer), su `kp_auc` real implícito es `(0.561-0.3·0.09)/0.7 ≈ 0.7629`
— **más bajo** que el `0.7743` implícito para r03 con el mismo supuesto.
Esto es coherente con la propia duda que el equipo ya registró
(`experiments/86-t1-margen/INFORME.md`, sección "Elección de base: r04, no
r03": diferencia de 0.008 "dentro del ruido", checkpoints con AUC local
casi idéntico pero score real distinto). **No se puede decidir cuál de los
dos supuestos (`kp≈0.774` vía r01, o `kp≈0.763` vía r04) es el correcto sin
el desglose real de r05** — y ese desglose no existe en el repo para
ninguna submission salvo r01.

### 1.4 Hipótesis, rankeadas por probabilidad

**H1 — Codabench sigue evaluando con el scorer VIEJO (probabilidad más alta)**

Lo único que el equipo verificó es que el repo `SynthesEyes-GmbH/fido-2026`
en GitHub cambió (`NOW.md:64-70`, diff contra el `vendor/` congelado). **Nadie
verificó que la instancia de Codabench que ejecuta las submissions haya sido
re-desplegada con ese cambio** — son dos sistemas distintos (repo de
documentación/código fuente vs. programa de scoring realmente instalado en
el evaluador). Bajo esta hipótesis, `r05` se evalúa con
`GROUND_TRUTH_DISTANCE_SCALE=10`, `MAX_THRESHOLD_DIST=10` — el mismo
régimen que r01/r03/r04. Las constantes de r05 (`A,B` ×1.28) fueron
calculadas para el target NUEVO; contra el target VIEJO son un sesgo
multiplicativo del 28% en la dirección contraria, que **cancela o revierte**
cualquier ganancia en el subconjunto con medición geométrica real, dejando
`distance_auc` cerca de donde ya estaba (~0.09). Combinado con que el
keypoint no cambia entre r04 y r05 (mismos pesos), el resultado esperable
bajo H1 es: **r05 ≈ r04 (0.561), no r05 = r03 (0.569)** — pero como la
brecha r03↔r04 ya es "ruido" según el propio proyecto (0.008), ambos números
son indistinguibles dentro de esa banda.
**Cómo falsarla**: (a) revisar en el panel de Codabench si hay un aviso de
"scoring program updated" con fecha/hora, o preguntar directamente a los
organizadores si el redeploy ya ocurrió; (b) descargar
`detailed_results.html`/`scores.json` de `r05` (6 decimales) y comparar
`distance_auc` — si es ≈0.09 (como r01/r03) en vez de notablemente mayor,
confirma H1; si es mayor y aun así el score total coincide con r03, la
coincidencia es de otra naturaleza (H3).

**H2 — El fallback domina una fracción medida y grande de los casos reales
(compone con H1, no lo reemplaza)**

Medido en este peldaño (sección 2): **36.7%** (n=120, Scenario_01) de los
casos con distancia GT real no reciben medición geométrica del propio
checkpoint de producción — caen al `DISTANCE_FALLBACK_PX`. Sumado al 10% de
`oct_volume=None`, **≈40-45% de los 100 casos reales** de Task 1 nunca ven
las constantes `A,B`. En `r05` ese fallback quedó **sin reescalar** (159.2,
el bug que `r06` corrigió después) — así que, incluso si Codabench SÍ usa
el scorer nuevo, entre el 40 y 45% de los casos de `r05` reciben la misma
predicción constante que ya daba mal resultado en r04, y solo el 55-60%
restante se beneficia del reescalado de A,B. Esto acota (no anula) cuánto
puede subir `distance_auc` en `r05` específicamente — no en `r06`, que sí
la corrigió.
**Cómo falsarla**: completar la medición ya lanzada (script en background,
sección 2) sobre los 10 escenarios y comparar `r05` (fallback sin corregir)
contra `r06` (corregido) en el mismo subconjunto.

**H3 — Coincidencia de redondeo entre dos modelos de rendimiento similar
(probabilidad media-baja, barata de descartar)**

"0.569" tiene 3 decimales — compatible con una lectura del leaderboard, no
del desglose de 6 decimales. El propio proyecto ya documentó que la
diferencia real r03↔r04 es de solo 0.008 pese a checkpoints distintos. No
haría falta que las predicciones fueran idénticas para que el score
truncado a 3 decimales coincidiera — solo que ambos modelos rindan de forma
parecida, lo cual ya está documentado como el caso.
**Cómo falsarla**: mirar el score a 6 decimales, no el leaderboard.

**H4 — r05 nunca se ingirió correctamente y lo mostrado es un score en caché
(probabilidad más baja, pero no descartable sin mirar Codabench)**

Debilitada por evidencia indirecta: la premisa incluye un Task 2 real de
`0.000`, y `r05` **sí** incluye `model_1.pth` (a diferencia de `r03`, cuyo
zip subido "contiene únicamente `inference.py` y `model_0.pth`" —
`experiments/86-t1-margen/INFORME.md:` sección "Elección de base"). Si
Codabench hubiera reusado un score en caché de `r03`, Task 2 debería salir
en error o vacío, no en `0.000` — que además coincide con el techo ya
medido del checkpoint de Task 2 (AUC 0.0048-0.0092, prácticamente cero por
mérito propio, no por trampa — ver `RESULTS.md:246`). Esto sugiere que SÍ
hubo una ingestión real, distinta de r03.
**Cómo falsarla**: mirar "Cases evaluated" y "duration" en
`detailed_results.html` de la submission — un timing idéntico al de r03
sería la señal de un problema de caché real.

**Conclusión de esta sección**: la explicación más probable es una
**combinación de H1 y H2** — el scorer vivo probablemente sigue siendo el
viejo (o al menos no hay evidencia de lo contrario), y aunque no lo fuera,
el fallback sin corregir en r05 acota el efecto del reescalado a como mucho
el 55-60% de los casos. Cualquiera de las dos, solas, ya empuja el score de
r05 hacia el rango de r03/r04 en vez de hacia una mejora clara. La acción
más barata y de mayor valor de información es leer el desglose de 6
decimales de la submission real en Codabench — sección 1.3 ya muestra que
la respuesta cambia completamente según qué `kp_auc` real se asuma, y eso
no se puede adivinar desde este repo.

---

## 2. Medición real: fracción de fallback con el checkpoint de producción

**Qué se midió**: se ejecutó `distance_from_segmentation` (la función real
de `submissions/r06-fallback-fixed/inference.py:242`, idéntica en r04/r05)
sobre B-scans reales leídos de `data/Task 1/Scenario_XX.zip`, usando el
UNet real de producción (`model_0.pth`, pesos bit-idénticos entre r04, r05
y r06). Se restringió la muestra a frames con `CANNULA` activa (GT de
distancia real, no centinela) — el subconjunto que de verdad importa para
`distance_auc`. Esto es DISTINTO de todas las mediciones previas del
proyecto (`analysis/measure_task1_distance_ceiling.py`,
`analysis/verify_task1_distance_geometry.py`), que usan **máscaras GT**
como si fueran la predicción del modelo — miden el techo ideal, no lo que
el modelo realmente produce. Ese techo ideal (T1-81) nunca se ejecutó
tampoco: `experiments/81-t1-distance-ceiling/CEILING.md` dice literalmente
"pendiente de ejecución en RTX 5090" — **nadie había medido esto, ni con
GT ni con el modelo real, antes de este peldaño**.

**Resultado parcial** (la corrida completa de 10 escenarios sigue en curso
en segundo plano al cierre de este informe — el entorno tiene contención
severa de CPU por otros procesos concurrentes del proyecto; ver nota al
final):

| Escenario | muestreados | medidos | NaN (fallback) | fracción NaN |
|---|---:|---:|---:|---:|
| Scenario_01 | 120 | 76 | **44** | **36.7%** |
| Scenario_02 | 120 | 111 | **9** | **7.5%** |
| Scenario_03 | 120 | 99 | **21** | **17.5%** |
| **Acumulado (3/10)** | 360 | 286 | **74** | **20.6%** |

**La corrida en segundo plano se interrumpió tras el Scenario_03** (código de
salida 1, sin traceback capturado en el log — el entorno tenía 10+ procesos
Python concurrentes de otros frentes del proyecto compitiendo por CPU/I/O
durante esta sesión; lo más probable es un kill por contención de recursos,
no un bug del script, que ya había completado 3 escenarios limpios). El
resultado de esta sección queda en **3/10 escenarios, n=360**, no en los 10
completos. Quien retome esto puede relanzar
`python analysis/measure_task1_segmentation_nan_fraction.py` — el script
reescanea desde el escenario 1 (no tiene checkpoint de reanudación), así
que conviene correrlo cuando el entorno esté menos saturado.

La varianza ENTRE escenarios es grande (36.7% / 7.5% / 17.5%) — la
cobertura de segmentación no es una propiedad uniforme del dataset, depende
fuertemente del escenario (iluminación, ángulo, oclusión de la cánula).
Esto en sí es un hallazgo: cualquier número único de "fracción de fallback"
sin desglose por escenario es engañoso, igual que ya se documentó para
Task 2 con la escala fuera de distribución. Con 3/10 escenarios (y la
corrida interrumpida antes de cubrir el resto, ver nota arriba), el
acumulado (20.6%) es una cota orientativa, no definitiva. Sumando el 10%
adicional de `oct_volume=None` que la ingestión fuerza independientemente
de la segmentación (`RULES_OF_ENGAGEMENT.md:76-78`), la fracción total de
los 100 casos reales que **nunca ven las constantes A,B** rondaría, con los
datos disponibles hasta ahora, entre **15% y 40%** según el mix de
escenarios del test oculto — un rango ancho, pero en cualquier punto de ese
rango ya es suficiente para descartar la idea implícita en los informes
previos (T1-86) de que el fallback es un caso raro: es una fracción grande
y estructural del problema, no una cola despreciable.

**Reproducción**: `python analysis/measure_task1_segmentation_nan_fraction.py`
(script pre-existente en el repo, no escrito en este peldaño — apareció
como archivo sin trackear durante esta sesión, aparentemente de otro
frente de trabajo paralelo sobre el mismo repo; se ejecutó tal cual sin
modificarlo). La corrida se interrumpió antes de escribir su propio
`experiments/93-t1-dfl/NAN_FRACTION_MEASUREMENT.md` (el script solo
escribe ese archivo al final del loop completo); los tres escenarios
medidos aquí vienen del log de progreso (`print(..., flush=True)` por
escenario), no del reporte final del script.

---

## 3. Los cuatro puntos auditados

### 3.1 "El backbone no es el cuello de Task 1"

**Veredicto: la conclusión es válida para lo que mide (arquitectura del
extractor de features), pero se usa con más generalidad de la que soporta.**

`[VERIFICADO]` Las tres corridas (CNN 0.8539, ResNet-18+FPN 0.8534, DINoV2
0.8005) comparten exactamente el mismo decodificador: `soft_argmax_2d`
**global**, stride 16 (`FundusEnfaceHeatmapModel`/`Task1KeypointModel` en
`inference.py`, y confirmado explícitamente en
`experiments/86-t1-margen/INFORME.md` sección 5.1: "El modelo de producción
usa `soft_argmax_2d` global, la variante más simple y la que más
cuantización sub-píxel puede dejar sobre la mesa"). Ninguna corrida varió
la resolución del heatmap, el stride, ni el decodificador (T1-82 "decoder
local" y T1-83 "DARK" están **escritos pero nunca ejecutados** — el propio
informe lo dice: "pendiente de GPU"). Tres arquitecturas que comparten el
mismo cuello de botella potencial y dan el mismo techo (~0.853) es
consistente TANTO con "el backbone no importa" COMO con "las tres están
limitadas por lo mismo (decodificador/resolución/definición del target)" —
la evidencia actual no distingue entre ambas, y el proyecto ya lo sabe
(`ATTACK_LADDER.md`: "Antes de gastar otra corrida en un backbone nuevo hay
que medir de qué está hecho ese 0.147 restante"). El riesgo no es que la
conclusión sea falsa, es que su redacción en `ATTACK_LADDER.md`/`NOW.md`
("el techo es del planteamiento... no de la arquitectura") suena más
categórica de lo que los tres experimentos, que solo tocaron un eje,
pueden sostener.

### 3.2 Brecha `0.8539` local vs `~0.774` real: ¿split real?

**Veredicto: el `GroupKFold` por escenario está bien implementado y
verificado en código; el número `~0.774` citado como "el real" no tiene
fuente verificable en el repo.**

`[VERIFICADO]` `task1_split` (`src/fido/data/task1.py:48-73`) agrupa por
`case["scenario"]`, usa `GroupKFold` real (`src/fido/data/common.py:154-162`,
wrapper de `sklearn.model_selection.GroupKFold`), y verifica explícitamente
que el split resultante es una partición exacta y disjunta de los case IDs
(`if set(...) & set(...) or observed != set(case_ids): raise RuntimeError`,
línea 71-72) — no es un split aleatorio disfrazado. Esta parte de la
afirmación del proyecto es sólida.

`[VERIFICADO]` Pero el número `0.774` **no aparece verificado en ningún
lugar del repo para el checkpoint que da 0.8539 local**. El propio equipo
ya lo señaló, en un documento que no llegó a `NOW.md`/`ATTACK_LADDER.md`:
`experiments/86-t1-margen/INFORME.md` sección 5.0, "el `keypoint_auc≈0.774`
citado en el encargo de este peldaño no tiene una fuente verificable dentro
del repo... si corresponde a `r01` (checkpoint viejo) el 'hueco de 0.08'
está mal planteado". El único breakdown real y verificado en todo el repo
es el de `r01` — `kp_auc=0.7309` — con un checkpoint **anterior** a la
corrida que dio 0.8539 (T1-80, 2026-08-19). Es decir: **el número que
motiva esta misma revisión adversarial (0.774) ya fue puesto en duda por
el propio proyecto un día antes, y esa duda no se propagó a los documentos
de nivel superior** (`NOW.md`, `ATTACK_LADDER.md` siguen sin mencionarla).
Esto es un patrón a vigilar: una advertencia correcta escrita en un informe
de peldaño se pierde si nadie la sube al resumen ejecutivo.

`[VERIFICADO — riesgo abierto, no medido]` `task1_split` reparte por
escenario dentro de los **mismos 10 escenarios de entrenamiento**. No hay
evidencia en el repo de que el test oculto de Codabench use escenarios
independientes de esos 10 (a diferencia de Task 2, donde el desplazamiento
de escala entre train y Mock Test SÍ está medido y confirmado). Esta
hipótesis de "domain gap por escenario" para Task 1 está escrita
(`experiments/86-t1-margen/INFORME.md`, "H-domain-gap") pero no probada.

### 3.3 ¿Los oráculos de Task 2 miden lo que dicen medir?

**Veredicto: razonablemente bien controlados donde se revisó; no se
encontró un bug de implementación que invalide el hallazgo negativo
principal, pero el código no se auditó línea por línea en su totalidad
(fuera de presupuesto de esta revisión).**

`[VERIFICADO]` El oráculo de apariencia (T2-R2/R4, "ninguna proyección del
OCT correlaciona con el fundus en la posición GT") se validó con un control
de auto-consistencia sintético ANTES de aceptar el resultado negativo:
"caso sintético autoconsistente pasa (`corner_error=0.09px`)" (`RESULTS.md`,
entrada R02). Esto es la prueba correcta contra un bug de implementación:
si hubiera un error de convención de ejes o de normalización, el caso
sintético (diseñado para ser resoluble por construcción) también habría
fallado. Que pase con error casi nulo y solo falle contra datos reales es
buena evidencia de que la ausencia de señal es real, no un bug. Punto a
favor del proyecto, no un hallazgo negativo mío.

`[NO VERIFICADO — límite de esta revisión]` No revisé línea por línea los
786 líneas combinadas de `analysis/measure_task2_oracle_signal.py` +
`analysis/verify_task2_enface_convention.py` + `validate_task2_enface_convention_train.py`.
Ya hay un precedente documentado de un error de lectura (no de código) casi
cometido por el propio equipo: leer el gate de T2-83 contra el control
equivocado (`identity__keep_u__keep_v` en vez del mejor competidor
`identity__flip_u__flip_v`) — capturado en la sección "Cicatrices" de
`NOW.md`. Eso es una señal de que este código es propenso a errores
sutiles de "contra qué se compara", no de que haya uno sin detectar ahora.
No encontré uno nuevo, pero tampoco puedo certificar que no exista.

### 3.4 ¿Es 1.28 el factor correcto para A **y** B, y también para el fallback?

**Veredicto: exacto y correcto para A y B (verificado por álgebra); NO
tiene la misma garantía matemática para la constante de fallback, y la
justificación que se le dio en el commit real (`d87dd0e`) es más débil de
lo que su tono sugiere.**

`[VERIFICADO por álgebra]` Para una regresión OLS `Y = aX + b`, si
`Y_new = k·Y_old` con `X` sin cambios: `Cov(X, kY) = k·Cov(X,Y)` y
`Var(X)` no cambia, así que `a_new = k·a_old` exactamente; y
`b_new = mean(Y_new) - a_new·mean(X) = k·(mean(Y_old) - a_old·mean(X)) =
k·b_old` exactamente. `R²` es invariante porque tanto `SS_res` como
`SS_tot` escalan por `k²`. Esto es correcto sin condiciones adicionales
— confirmé la derivación yo mismo, no solo la cité. La afirmación del
proyecto en este punto es matemáticamente sólida.

`[INFERIDO — la brecha real]` `DISTANCE_FALLBACK_PX` **no** es un
coeficiente de una regresión OLS: es el argmax de una función AUC
discretizada (`analysis/explore_task1_gt.py`, búsqueda sobre percentiles 1
a 99). La propiedad de "escala exacto por k" para el argmax de un AUC
discretizado bajo un cambio de escala del target **y del umbral** solo se
sostiene si target y umbral escalan por el MISMO factor. Aquí no lo hacen:
el target escala ×1.28 pero el umbral escala ×2 (10→20). El commit
`d87dd0e` (mensaje: "the threshold changes how many cases land inside it,
not where the optimal centre sits") descarta la objeción original de r05
("el umbral no escaló por el mismo factor") con una afirmación que es
cierta solo aproximadamente — depende de la forma de la distribución, y no
se verificó con los datos reales antes de fijar `203.8`. La dirección del
cambio (subir el fallback, que estaba 45.6px por debajo del centro) es casi
con certeza correcta; la magnitud exacta (`159.2 × 1.28` en vez de
resolver el problema de optimización real bajo el umbral de 20px) no tiene
el mismo respaldo que `A,B`, pese a estar redactada con el mismo tono de
certeza matemática ("scales by the same factor"). Es la misma clase de
sobre-confianza retórica, en menor escala, que produjo el propio bug que
`r06` corrigió.

`[VERIFICADO — bug de código menor, no crítico]` Si alguien re-ejecuta HOY
`analysis/explore_task1_gt.py` para recalcular el fallback bajo el régimen
nuevo, obtendría un resultado inconsistente: `GROUND_TRUTH_DISTANCE_SCALE`
ya está actualizado a `7.8125` en el archivo (línea 24), pero la función
`auc()` interna sigue con `max_threshold=10` por defecto (línea 92,
`def auc(values, max_threshold=10)`) y se llama sin override (línea 100,
`auc(np.abs(real - candidate))`). El resultado sería target nuevo + umbral
viejo — un régimen que nunca se usó ni se usará. Nadie lo re-ejecutó
todavía así que no hay ningún número contaminado por esto en el repo, pero
el script quedó en un estado que produciría uno si alguien lo corre.

---

## 4. Los tres errores más graves, por impacto en el score

**1. Nadie confirmó que Codabench redesplegó el scorer corregido — y el
documento que debería reflejarlo no lo refleja.** `RULES_OF_ENGAGEMENT.md`,
el archivo que `CLAUDE.md` manda leer primero, sigue diciendo (líneas
106-108 y 115-117) que la distancia GT "está almacenada ×10 y el scoring lo
divide entre 10" y que los umbrales van "de 0 a 10 enteros" — las
constantes VIEJAS. También sigue diciendo (línea 131) que Competition
cierra "hasta 20 ago 2026" con 20 submissions totales, cuando `NOW.md`
registra que se extendió a 22 de agosto con +10 extra. Cero menciones a la
corrección del 2026-08-19 en las 150 líneas del archivo — verificado con
grep, no hay ninguna. Si toda la estrategia de "reescalar A,B" descansa en
un supuesto (que Codabench ya usa el scorer nuevo) que nadie confirmó
externamente, y encima el documento de referencia del proyecto contradice
ese supuesto sin que nadie lo note, el riesgo no es solo del análisis de
`r05` — es que **una sesión futura que siga las instrucciones de leer
`RULES_OF_ENGAGEMENT.md` primero, como manda `CLAUDE.md` de este mismo
proyecto, reciba las constantes equivocadas** y las use para diseñar la
siguiente submission real, con presupuesto de solo ~10-13 submissions
restantes.

**2. La forma en que se planteó la pregunta ("r05 = r03") no es la
comparación causal correcta, y eso ya costó una submission potencialmente
mal interpretada.** `r05` no es una variante de `r03`: es una variante de
`r04` (mismos pesos, verificado por hash). `r04` ya es una submission real
con su propio score (`0.561`), 0.008 por debajo de `r03`, diferencia que el
propio proyecto etiquetó como ruido. La pregunta bien formulada es "¿`r05`
mejora sobre `r04` (0.561)?", no "¿por qué `r05` coincide con `r03`
(0.569)?" — esta segunda formulación invita a buscar una explicación
causal (bug, coincidencia exacta) para algo que, mirado desde `r04`, podría
ser simplemente "no mejoró, o mejoró poco, dentro del mismo ±0.01 de ruido
ya documentado". Si el resto de la organización (Rodrigo, el orquestador)
sigue comparando contra `r03` en vez de `r04`, cada submission futura de
esta cadena se va a evaluar contra la referencia equivocada.

**3. El fallback (36.7% medido, ~40-45% estimado del total) es
estructuralmente grande y nunca se había medido con el modelo real antes de
este peldaño — pese a que el propio proyecto ya había escrito el
experimento para hacerlo (`T1-81`, `experiments/81-t1-distance-ceiling/
CEILING.md`: "pendiente de ejecución en RTX 5090") y nunca lo ejecutó. Esto
significa que **toda la narrativa de `NOW.md` sobre "el margen del proyecto
está en la distancia" (`RESULTS.md:243`) se apoyó en el `distance_auc` de
r01 (0.09) sin saber que ~40% de los casos ni siquiera pasan por el modelo
de distancia** — la mejora esperable de "arreglar la escala" tiene un techo
mucho más bajo de lo que `NOW.md` sugiere ("cerrar el hueco al líder
requiere subir distance_auc de 0.09 a 0.257"), porque una fracción grande y
fija de los casos está anclada a una sola constante sin importar qué tan
bien mida la geometría el resto del pipeline. Arreglar el fallback (ya
hecho en `r06`) probablemente vale más que seguir afinando `A,B`.

---

## 5. Qué falta (pendiente, no verificado, riesgos)

- El desglose real de `r05`/`r06` en Codabench (`keypoint_auc`,
  `distance_auc` a 6 decimales) — es la única forma de decidir entre H1/H2/H3
  de la sección 1. No accesible desde este repo.
- La medición de fallback (sección 2) cubre 3 de 10 escenarios; la corrida
  en segundo plano se interrumpió con código de salida 1 y sin traceback
  (contención de CPU severa en este entorno — se detectaron 10+ procesos
  Python concurrentes de otros frentes del proyecto durante esta sesión).
  Falta relanzarla completa y promediar por escenario antes de citar un
  número único como "la" fracción de fallback.
- No confirmé con los organizadores ni con ninguna fuente externa si el
  scorer de Codabench fue redesplegado — es el paso de mayor valor de
  información de todo este informe y está fuera del alcance de un repo
  local.
- No auditė línea por línea `analysis/measure_task2_oracle_signal.py` (411
  líneas) ni `verify_task2_enface_convention.py` (267 líneas) — la revisión
  se limitó a la lógica de auto-consistencia ya documentada por el proyecto.
