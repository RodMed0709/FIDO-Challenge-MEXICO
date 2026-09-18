# T2-86 — Postmortem del 0.00 en Task 2 de `r04`, pregunta DINOv2/LP-FT/ResNet, y siguiente peldaño

**Fecha**: 2026-08-19. **Autor**: agente de investigación (sin escritura de código de producción).
**Punto de partida**: las tres evidencias de que la vía de apariencia de Task 2 está muerta
(oráculo negativo sobre 1214 casos, T2-83 vasculatura plana en 8 orientaciones, T2-82
shuffle-drop en ruido) se dan por asentadas, no se re-litigan.

**Convención de lectura**: cada afirmación factual lleva `archivo:línea` como evidencia.
Donde razono en vez de medir, digo explícitamente "inferencia" o "no medido".

---

## Resumen ejecutivo

1. **El 0.00 de `r04` en Task 2 NO es una trampa de contrato/formato.** Es un score real,
   ya reproducido **localmente antes de la submission** con el mismo código de scoring
   vendorizado que usa Codabench. `r04` cumple el contrato punto por punto (firma, nombres
   de archivo, forma de la matriz devuelta, manejo de `oct_volume=None`, excepción global
   capturada). El AUC es 0.000000 porque, con el modelo que hay, ningún caso cae por debajo
   de 10 px de error de esquina — algo ya esperado a partir de su propio techo de
   entrenamiento (AUC 0.0048–0.0092) y agravado por el desplazamiento de escala fuera de
   distribución.

2. **DINOv2 en Task 2 SÍ estaba frozen por defecto** (confirmado en código), así que la
   objeción de Rodrigo es técnicamente correcta sobre esa corrida puntual. Pero LP→FT no
   ataca el cuello real: T2-82 ya probó una CNN **completamente entrenable** (no frozen) y
   colapsó sin usar el OCT, y T2-83 midió que ni siquiera el oráculo (features perfectas,
   sin red de por medio) encuentra correlación en la vasculatura. Congelar/descongelar
   backbone es un eje de capacidad; el problema medido es ausencia de señal. LP→FT (T2-85)
   está pre-registrado pero **nunca se ejecutó**.

3. **Siguiente peldaño propuesto para Task 2**: no otra vía de apariencia (ni aprendida ni
   clásica — ambas dependen de la misma señal ya falsada), sino una comprobación barata y
   sin GPU de si la posición de Task 2 correlaciona con algo que **no sea apariencia**
   (el keypoint de Task 1, u otro campo no-visual). Es la única vía viva que ningún
   experimento anterior tocó. Criterio de aceptación pre-registrado abajo.

---

## (a) Diagnóstico del 0.00 en Task 2 de `r04`

### Contexto verificado primero

`r04` obtuvo en Codabench `Task 1 = 0.561`, `Task 2 = 0.00` (`ATTACK_LADDER.md:243`).
El mismo zip, la misma corrida, el mismo contrato — Task 1 puntuó con normalidad y Task 2
dio cero exacto. Esto ya es una pista fuerte: una trampa de contrato (nombre de archivo,
firma, excepción no capturada) suele tumbar la corrida **completa**, no una sola tarea,
porque ambas comparten `inference.py` e ingestion.

### Hallazgo decisivo: el 0.00 ya se había reproducido localmente, con el scorer real, antes de subir

- `eval/run_local.py` importa los módulos de ingestión y scoring del bundle oficial
  **sin modificarlos** — solo reescribe rutas hardcodeadas de `/app/...`
  (`eval/run_local.py:4-5,41,54-55,66-67`).
- `submissions/r04-interim-joint/local_score_task2.json:6-16` — sobre los 5 casos del Mock
  Test: `"final_score": 0.0`, `"num_cases": 5`, `"timed_out": 0`, sin campo `"error"`. Es
  decir, el pipeline de scoring **corrió hasta el final sin excepción** (compárese con
  `scoring_registration.py:290-297`, que solo escribe `"error"` si algo revienta) y calculó
  un AUC real de 0.
- `submissions/r04-interim-joint/README.md:27-29` documenta esto **antes de la submission**:
  *"Task 2: `0.000000`, 0 timeouts. La mejora del holdout de escala no se transfiere al Mock
  fuera de distribución y no se considera un avance de leaderboard."* El equipo ya sabía
  que Task 2 daba cero localmente y subió de todos modos (para medir Task 1, según
  `submissions/r04-interim-joint/README.md:1-4`).

Esto es casi tan cerca de un smoking gun como se puede pedir: el mecanismo de scoring real
(no la documentación, no una reimplementación) ya daba 0.00 antes de tocar Codabench, y
Codabench devolvió el mismo 0.00. Coincidencia exacta entre dos ejecuciones del mismo
modelo contra el mismo scorer, en dos datasets distintos (Mock local vs. test oculto).

### Hipótesis rankeadas

**H1 — Fallo genuino del modelo (no hay señal explotable) + escala fuera de distribución. Probabilidad: alta, prácticamente confirmada.**

Mecanismo verificado en `scoring_registration.py:91-98` (`auc_from_errors`): el AUC es el
promedio de `accuracy@threshold` para umbrales enteros 0..10 px. Si **ni un solo caso**
de los evaluados cae en `error <= 10`, el resultado es exactamente `0.000000` — no "cerca
de cero", literalmente cero, porque cada término de la suma es cero.

Esto es consistente con lo ya medido sobre el propio fold de entrenamiento del checkpoint
que usa `r04` (T2-81, época 38): `val_corner_auc=0.0092`, error medio 77.60 px sobre 248
casos (`RESULTS.md:90`, `ATTACK_LADDER.md:625-629`). Un AUC de 0.0092 sobre 11 umbrales
implica una tasa de acierto a 10 px ya muy baja **incluso en la distribución de
entrenamiento**. Fuera de distribución (Mock/test oculto) es peor: `NOW.md` documenta que
la escala de train `[126.66, 188.09]` no solapa con la del Mock Test `[213.81, 226.16]`
(`NOW.md`, sección "Riesgo abierto — escala de Task 2 fuera de distribución"), y el rango
de augmentación de T2-81 (`[0.819208, 1.220691]` sobre el rango de train,
`ATTACK_LADDER.md:612-614`) cubre como mucho hasta ≈176×1.22≈215 px — el borde inferior del
rango del Mock, no su centro. Con esto, que ninguno de los 100 casos oficiales (muestra fija
determinista, semilla `2026`, `ingestion_registration.py:22,99-122`) caiga bajo 10 px es
plausible sin invocar ningún bug: no es una muestra mala, es la muestra completa y fija.

*Inferencia (no medida)*: si la tasa real de acierto a 10 px en régimen OOD es del orden de
≤1%, la probabilidad de cero aciertos en 100 casos independientes es `(0.99)^100 ≈ 37%` —
nada extraordinario. Si fuera del orden de 5% (la tasa aparente en el fold de entrenamiento
in-distribution), sería más raro (`(0.95)^100 ≈ 0.6%`), lo que refuerza que el régimen
fuera de distribución del test oculto es peor que el de entrenamiento, no solo "más de lo
mismo". No tengo el corner-error real de los 100 casos de Codabench para verificar esta
tasa — es un argumento de plausibilidad, no una medición.

**Cómo se falsaría**: correr `r04/inference.py::_infer_task2` sobre los 1214 casos de train
(o un fold de validación amplio) y medir la distribución completa de `corner_error`; si el
percentil bajo ya está muy por encima de 10 px incluso in-distribution, H1 queda confirmada
sin apelar a OOD. (No se hizo en este postmortem — es la verificación natural si se quiere
cerrar el caso al 100%, cuesta minutos de CPU con los datos ya locales.)

**H2 — Trampa de contrato/formato silenciosa. Probabilidad: descartada.**

Verificado punto por punto contra `RULES_OF_ENGAGEMENT.md` y el código vendorizado:

| Punto del contrato | Evidencia en `r04` |
|---|---|
| Firma `inference(task_id, oct_volume, opmi_image, model)` | `inference.py:455` — exacta |
| `model_1.pth` en la raíz del zip | Verificado con `zipfile.namelist()`: `inference.py`, `model_0.pth`, `model_1.pth`, plano, sin carpeta anidada |
| Devuelve `(3,3)` float64 con fila inferior `[0,0,1]` | `_compose_similarity` (`inference.py:318-328`); `validate_prediction` (`ingestion_registration.py:145-152`) solo exige forma `(3,3)`, y se cumple |
| Maneja `oct_volume=None` sin excepción | `_infer_task2` (`inference.py:434-437`) — rama explícita con fallback determinista |
| Ninguna excepción escapa de `inference()` | `inference.py:462-474` — `try/except` global con fallback numérico |
| `requirements.txt` no declara `numpy`/`torch` | No está presente en el zip (no es obligatorio) |

Y el argumento decisivo: el mismo código, pasado por el mismo scorer vendorizado sin
modificar, **ya dio 0.00 localmente sin ningún `"error"` en `scores.json`** (ver hallazgo
anterior). Si hubiera un bug de contrato que aborta la ingestión o la predicción, el
`scoring_registration.py:290-297` lo habría capturado como `"error": str(exc)`, no como
un `final_score: 0.0` limpio con `num_cases: 5` y `timed_out: 0`. Task 1, en el mismo zip,
con el mismo mecanismo de carga (`load_model`, `inference.py:364-388`), puntuó con
normalidad tanto local (`0.625455`) como en Codabench (`0.561`) — otra señal de que la
ingestión del zip funciona.

**H3 — Convención en-face incorrecta horneada en el checkpoint desplegado. Probabilidad:
descartada como causa del 0.00 específicamente (aunque real como limitación del modelo).**

`src/fido/data/common.py:112` define `TASK2_ENFACE_CONVENTION = "transpose__flip_u__flip_v"`
(la ganadora de T2-83, `ATTACK_LADDER.md:676`), con una función de aplicación en
`common.py:121-130`. Pero el checkpoint que usa `r04` (`task2_scale_aug`, entrenado en T2-81,
"LOGRADO" el 2026-08-18 — un día **antes** de T2-83) y su réplica autocontenida en
`_enface_from_volume` (`submissions/r04-interim-joint/inference.py:348-357`) **no aplican
ningún flip/transpose**, solo `volume.mean(axis=1)`. Esto es consistente train/inferencia
(el modelo se entrenó con la convención vieja y `inference.py` la replica fielmente,
cumpliendo su propia regla de "AUTOCONTENIDO... o los pesos dejarán de casar",
`inference.py:8-11`) — **no es un bug de submission**, es una limitación de qué checkpoint
se empaquetó. No explica por qué el score es *exactamente* 0.00 en vez de "bajo pero no
cero"; para eso basta H1. Vale la pena anotarlo porque cualquier reentrenamiento futuro de
Task 2 debería usar la convención correcta, pero no cambia el diagnóstico de esta
submission.

### Conclusión de (a)

El 0.00 de `r04` en Task 2 es un resultado real del modelo, no una trampa de formato.
Fue reproducido con el scorer oficial antes de subir, documentado en el propio README de
la submission, y es consistente con el techo ya medido del checkpoint (AUC 0.0048–0.0092
in-distribution) agravado por el vacío de escala fuera de distribución. Investigar más
"por qué 0.00 y no 0.02" tiene retorno bajo: el problema de fondo — sin señal de apariencia
explotable — ya está establecido por las tres evidencias que este informe no re-litiga.

---

## (b) DINOv2 frozen, LP→FT y la alternativa ResNet-18

### Lo que se verificó

- `src/fido/models/task2_dinov2.py:19,31-33` — `FundusEnfaceDinoV2Model.__init__` tiene
  `freeze_backbone: bool = True` por defecto; si es `True`, hace
  `self.backbone.requires_grad_(False)` y `self.backbone.eval()` (línea 32-33), y
  `train()` (líneas 51-57) fuerza al backbone a seguir en `eval()` aunque el resto del
  modelo esté en modo entrenamiento — es decir, el backbone queda genuinamente congelado,
  no solo "con LR bajo".
- `src/fido/train/train_task2_baseline.py:122,215` — el script de entrenamiento expone
  `--freeze-backbone` como `BooleanOptionalAction` con `default=True`, y lo pasa directo
  al constructor. No encontré en el repo un log o comando que documente haber pasado
  `--no-freeze-backbone` para la corrida que produjo `task2_dinov2/model_1.pth`
  (`HANDOFF_2026-08-18.md:319-323`, `experiments/82-t2-common-cnn/train_run.log:323`). Dado
  que el flag por defecto es `True` y no hay evidencia de override, **la corrida medida
  (`val_corner_auc=0.0000` en épocas 30/35/40, `RESULTS.md:105`) casi con certeza usó el
  backbone congelado.** Esto no lo verifiqué leyendo el comando exacto invocado (no lo
  encontré en el repo) — es inferencia a partir del valor por defecto del código, con
  confianza alta pero no absoluta.
- `experiments/85-t2-dino-lpft/RESULTS.md:1-4` — **T2-85 (LP→FT) está pre-registrado
  (`PRE_REGISTRATION.md`) pero NUNCA se ejecutó**: "PRE-REGISTRADO Y BLOQUEADO... No se
  ejecutó GPU ni Mock." Su propia condición de apertura (`PRE_REGISTRATION.md:3-6`) exige
  gate T2-83 en verde (ya lo está) y un T2-82 repetido bajo la convención canónica (no
  hecho). Es decir: la pregunta de Rodrigo apunta a un experimento real que existe en el
  repo, pero que el equipo bloqueó a propósito y no corrió.

### Respuesta a la pregunta de Rodrigo

**Sí, la objeción es correcta en un punto concreto**: el DINOv2 que perdió en Task 2 (AUC
0.0000, divergiendo de 103 a 137 px en vez de mejorar) estaba con el backbone congelado por
defecto. No es una prueba justa de "cuánta señal puede extraer DINOv2 si se le permite
adaptarse al dominio OCT/fundus", que es visualmente muy distinto de las imágenes naturales
con las que se preentrenó.

**Pero eso no significa que LP→FT vaya a arreglar Task 2**, y aquí está el porqué, con la
distinción que Rodrigo pidió — qué mide cada evidencia y qué no:

| Evidencia | Qué backbone usó | Qué mide | Resultado |
|---|---|---|---|
| Oráculo de apariencia (`ORACLE_FULL.md`, 1214 casos) | Ninguno — correlación cruda (NCC) sobre las proyecciones reales | Si el contenido de imagen en la posición GT exacta se parece al fundus, **sin ningún modelo de por medio** | Negativo |
| T2-83 (`oracle_train.json`, 966 casos train) | Ninguno — mismo tipo de oráculo, con la convención en-face ya corregida | Igual que arriba, descartando "convención mal" como causa | Vasculatura plana (ratios 0.81–1.04) en las 8 orientaciones |
| T2-82 (descriptores comunes) | **CNN desde cero, completamente entrenable** (no congelada) — objetivo contrastivo InfoNCE | Si una red que SÍ puede adaptar todos sus pesos, con capacidad de aprender cualquier mapeo no lineal, encuentra una representación conjunta útil | Colapsó a solución trivial; `shuffle_drop_points` en ruido puro alrededor de 0 (`ATTACK_LADDER.md:642-660`) |
| DINOv2 frozen (A/B de backbone) | ViT-S/14 preentrenado, **congelado** | Si features genéricas de visión natural, sin adaptar, sirven de zero-shot | Divergió, AUC 0.0000 |
| T2-85 LP→FT (propuesto por Rodrigo) | ViT-S/14 preentrenado + 4 últimos bloques finos | Si adaptar parcialmente un backbone preentrenado (no desde cero, no completamente congelado) encuentra algo que ni el oráculo ni la CNN entrenable encontraron | **No corrido** |

El punto central: **T2-82 ya es la versión "sin restricciones de capacidad" de este
experimento.** Una CNN entrenada de cero, con todos sus parámetros libres, optimizando
directamente un objetivo de correspondencia (InfoNCE), tuvo more grados de libertad que un
DINOv2 con solo 4 bloques descongelados y LR de `1e-5` en el último
(`PRE_REGISTRATION.md`, sección "Arquitectura y fases congeladas") — y aun así colapsó sin
usar el OCT. Si el eje que faltaba fuera "capacidad de adaptación", T2-82 debería haber
encontrado *algo*, aunque fuera débil. No encontró nada: el colapso del `train_loss` a
`3.7e-7` (`ATTACK_LADDER.md:652-653`) es la firma de un objetivo contrastivo que encuentra
un atajo trivial (o ninguna diferencia entre positivos y negativos), no de underfitting.

Y el argumento más fuerte no depende de ninguna red: **T2-83 mide directamente si la
correspondencia existe en los datos**, sin backbone, sin optimización, en la posición GT
exacta (el caso más favorable posible para cualquier método). Si ni una comparación cruda de
densidad vascular en la ubicación correcta encuentra señal (ratios pegados a 1.0 en las 8
orientaciones), ningún backbone —por rico que sea su preentrenamiento— puede recuperar una
correlación que no está en los píxeles.

**Lo que LP→FT SÍ mediría que las tres evidencias no miden** (para ser honesto en el otro
sentido, como pide la tarea): las tres evidencias usan (a) estadística lineal/no-aprendida
(NCC, ratios de densidad) o (b) una CNN entrenada desde cero con **solo 966–1214 ejemplos**.
Es concebible —no medido, pura hipótesis— que exista una correlación **no lineal y sutil**
que ni la estadística cruda detecta ni una CNN pequeña entrenada desde tan pocos datos puede
aprender por sí sola, pero que sí emerja al usar representaciones preentrenadas en millones
de imágenes naturales (curvas, texturas, bordes a múltiples escalas) como prior. Este es
exactamente el tipo de señal que un LP→FT real intentaría capturar y que ningún experimento
anterior probó. El riesgo simétrico: con solo 966 pares de entrenamiento y un objetivo
contrastivo, un ViT parcialmente fino tiene más superficie para memorizar atajos espurios
(igual que T2-82) sin que eso generalice — el mismo modo de fallo, con un backbone más caro.

**Recomendación honesta**: LP→FT no es descabellado como pregunta científica, pero **no es
la apuesta de mayor retorno esperado dado lo ya medido**, porque ataca un eje (capacidad
para explotar el contenido de imagen) que T2-82 ya probó sin restricción de arquitectura
desde-cero y T2-83 ya cerró a nivel de oráculo. Yo no lo priorizaría antes que el peldaño
de la sección (c). Si de todos modos se quiere zanjar la pregunta de Rodrigo con evidencia
directa y barata, la única versión que aporta información nueva **real** es correr
únicamente la fase LP (3 épocas, backbones congelados, sin el costo de FT) de T2-85 y mirar
`shuffle_drop_points`: si incluso con features DINOv2 ricas el LP da el mismo patrón de
ruido que T2-82, es una cuarta confirmación independiente y cierra la pregunta de una vez.
No lo ejecuté — hace falta GPU y no es mi mandato en esta tarea.

**Sobre ResNet-18 como alternativa**: no aporta nada distinto de lo ya medido en Task 1
(`RESULTS.md:98-108`: CNN desde cero 0.8539 ≈ ResNet-18+FPN 0.8534, mismo techo con tres
arquitecturas) y Task 2 (CNN desde cero ganó a DINOv2 frozen, `RESULTS.md:104-105`).
Cambiar el backbone de "CNN desde cero" a "ResNet-18" en Task 2 sería repetir, con una
cuarta arquitectura, un eje que ya se demostró plano dos veces (Task 1 con tres backbones,
Task 2 con dos). No hay expectativa de que mueva la aguja, y no hay evidencia que lo
sugiera.

---

## (c) Siguiente peldaño propuesto para Task 2

### Por qué no otra vía de apariencia

Tanto T2-R4 Motor 2 (registro aprendido sobre probabilidades de vaso) como T2-R5
(DPCN++/correlación de fase log-polar) — las dos vías "geométricas" que
`ATTACK_LADDER.md` marca como vivas (`ATTACK_LADDER.md:339-342,443-456`) — dependen de la
**misma señal de apariencia** que las cuatro evidencias ya cerraron: ambas necesitan que el
contenido visual de la proyección en-face del OCT se parezca, en alguna representación, al
fundus en la posición correcta. T2-R4 Motor 2 lo dice explícitamente
(`ATTACK_LADDER.md:341-342`: "depende de resolver primero el hallazgo sobre
`enface_vessel_density`"), y ese hallazgo nunca se resolvió — se confirmó negativo cuatro
veces (oráculo T2-R11/R11b, T2-83, T2-82, y el oráculo específico de T2-R4 con NCC≈0 en los
4 casos válidos del Mock Test, `ATTACK_LADDER.md:349-360`). DPCN++ es correlación de fase
sobre el mismo par de imágenes — si no hay correspondencia de contenido, la correlación de
fase tampoco tiene nada que encontrar; es una técnica más robusta para *extraer* una señal
débil, no para inventar una que no existe.

Proponer T2-R4/R5 ahora mismo repetiría, con otra técnica, un experimento cuyo insumo
crítico ya se midió ausente. Es la misma decisión que el usuario ya tomó para T2-82/T2-85:
no vale la pena.

### Peldaño propuesto: T2-R12 — correlación no-visual entre posición de Task 1 y traslación de Task 2

**Por qué esta y no otra**: T2-80 mide que la posición `(tx,ty)` explica el 65.6% del error
del modelo actual (`ATTACK_LADDER.md:571-573`), muy por encima de rotación (12.5%) y escala
(5.4%). Es el componente que más vale atacar. Todo lo intentado hasta ahora para resolver
posición pasa por **contenido de imagen** (heatmap de correlación cruzada, template
matching, descriptores comunes). Ninguna evidencia negativa existente descarta que la
posición correlacione con algo **no visual**: la ubicación del instrumento en el fundus
(Task 1), el escenario, o el índice del frame dentro de la secuencia — señales que ya están
en el dataset y no requieren que el en-face "se parezca" a nada.

**Hipótesis**: durante una intervención quirúrgica real, el operador centra el volumen OCT
cerca de donde está trabajando con el instrumento. Si el simulador reproduce ese
comportamiento, el keypoint de Task 1 (punta del instrumento en coordenadas de fundus) y el
centro de la región registrada por Task 2 (`(tx,ty)` de la matriz GT, o el centro
proyectado de las 4 esquinas) deberían estar correlacionados — no perfectamente, pero más
que ruido.

**Pre-registro**:
- Script: cargar, para cada caso con Task 1 y Task 2 en el mismo frame (mismo
  `Scenario_XX/frame_id`), el keypoint GT de Task 1 (`Ground Truth`/instrumento) y el centro
  del cuadrado unitario proyectado por la matriz GT de Task 2 (promedio de las 4 esquinas
  proyectadas, mismo cálculo que usa `scoring_registration.py:47-54`).
- Medir correlación (Pearson y R² de una regresión lineal simple) entre la posición del
  keypoint de Task 1 y el centro de Task 2, por separado en x e y, sobre el split de train
  completo (evitar Mock, por la regla ya establecida de no seleccionar con Mock).
- **Costo**: CPU local, sin GPU, del orden de minutos — es lectura de JSON ya extraído
  localmente, no requiere pod.

**Criterio de aceptación (GO/NO-GO)**:
- **GO** si R² > 0.30 en x o en y (umbral orientativo: suficiente para que un predictor
  lineal barato reduzca el error medio de posición por debajo del que ya da el modelo
  actual, 22.80 px según T2-80 con posición GT vs. situación actual sin GT). Si GO: entrenar
  un predictor no-visual de `(tx,ty)` (regresión lineal o MLP pequeño sobre el keypoint de
  Task 1 + metadata) y medir su AUC de esquinas contra el mismo fold que T2-80, con
  rotación/escala del modelo actual (o valores medios) como controles fijos.
- **NO-GO** si R² < 0.05 en ambos ejes (el mismo orden de magnitud que el ruido ya medido en
  los oráculos de apariencia). Si NO-GO: es la quinta evidencia independiente de que Task 2
  no tiene, con las señales hoy disponibles, ningún eje explotable barato — visual o no
  visual — y la recomendación pasa a ser formal: cerrar la inversión activa en Task 2 hasta
  que cambien los datos disponibles, y mover el presupuesto de tiempo restante (3 días a
  cierre de Competition, `NOW.md`) a lo que ya tiene mayor retorno medido: la submission de
  distancia reescalada de Task 1 (`NOW.md`, "Lo siguiente, en orden", punto 1) y la
  ablation del error residual de Task 1 (punto 3).

**Qué NO es este peldaño**: no es una promesa de que Task 2 se arregla. Es la única
comprobación barata que queda que no repite un experimento ya hecho, y que separa con un
número — no con intuición — si vale la pena seguir invirtiendo en Task 2 esta semana o si
el proyecto debe aceptar formalmente que Task 2 se queda cerca del piso mientras el margen
real está en Task 1 (brecha de 0.050 al líder, ya cuantificada y accesible según
`ATTACK_LADDER.md:243`).

**Nota de honestidad**: incluso en el caso GO, el techo de este peldaño es limitado. T2-80
ya muestra que sustituir *solo* posición por GT (dejando rotación/escala del modelo actual)
da AUC 0.0480, muy por debajo del líder (0.475) y muy por debajo del techo de 4-DOF de la
parametrización (0.794, `RESULTS.md:37`). Un predictor de posición no-visual, con ruido
propio, dará menos que la posición GT perfecta. Este peldaño es un chequeo de si vale la
pena seguir en Task 2, no una vía hacia el podio en Task 2 por sí sola.

---

## Resumen de qué fue verificado vs. inferido

**Verificado leyendo código/artefactos**: contrato de `r04` completo contra
`RULES_OF_ENGAGEMENT.md`; reproducción local del 0.00 con el scorer vendorizado sin
modificar; mecanismo de `auc_from_errors`; `freeze_backbone=True` por defecto en
`task2_dinov2.py` y `train_task2_baseline.py`; T2-85 pre-registrado y nunca ejecutado;
convención en-face no aplicada en `r04/inference.py` (consistente train/serve, no es bug de
esta submission); estructura plana del zip de `r04`.

**Inferido (no medido en este postmortem)**: que la corrida de DINOv2 que dio AUC 0.0000
usó el flag por defecto (no encontré el comando exacto, solo el default del código y
ausencia de evidencia de override); el argumento de plausibilidad de "cero aciertos en 100
casos" a partir de tasas de acierto extrapoladas del fold de entrenamiento; el resultado del
peldaño T2-R12 propuesto, que no se ejecutó.
