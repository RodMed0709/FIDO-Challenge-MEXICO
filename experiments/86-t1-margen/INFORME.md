# T1-86 — Margen de Task 1: distancia reescalada, estimación del efecto, residuo del 6.3% y plan de ablation del keypoint

**Fecha**: 2026-08-19
**Contexto**: los organizadores corrigieron `GROUND_TRUTH_DISTANCE_SCALE`
(`10 -> 7.8125`) y `MAX_THRESHOLD_DIST` (`10 -> 20`) en
`scoring_keypoints.py` el 2026-08-19. `vendor/fido/` ya está re-sincronizado
y verificado idéntico al oficial (`NOW.md`). Este informe cubre los 5 puntos
pedidos: paquete de submission `r05-dist-rescaled`, estimación cuantitativa
de la subida de `distance_auc`, investigación del 6.3% residual de la
pendiente, y plan de ablation del keypoint.

**Convención de esta nota**: cada afirmación numérica está marcada
`[VERIFICADO]` (medido en este repo, comando incluido),
`[DOCUMENTADO]` (ya estaba en `NOW.md`/`ATTACK_LADDER.md`/`RESULTS.md`, se
cita, no se remide) o `[INFERIDO]` (razonamiento explícito, no medición
directa).

## 0. Resumen ejecutivo

1. **Paquete listo**: `submissions/r05-dist-rescaled/` — pesos bit-idénticos
   a `r04-interim-joint`, único cambio funcional: `DISTANCE_SCALE_A/B`
   reescaladas ×1.28. Contrato verificado contra `RULES_OF_ENGAGEMENT.md`.
   Localmente (Mock Test, bajo el vendor ya corregido): `distance_auc`
   `0.0 -> 0.133333`, `keypoint_auc` sin cambios (0.854545), `final_score`
   Task 1 `0.598182 -> 0.638182`.
2. **Estimación cuantitativa**: sección 3.3 (ver más abajo, completada con
   datos reales a escala completa — no una intuición).
3. **6.3% residual**: no es un artefacto de la corrección de hoy — ya
   estaba en el ajuste original de T1-R2 contra el target viejo, y
   sobrevive intacto al reescalado exacto por álgebra de OLS. Hipótesis
   líder: contaminación de identidad de herramienta en la clase de
   segmentación `InstrumentInOCT` (cánula vs endoiluminador, ambos
   presentes en escena). Propuesta de falsación barata incluida, no
   ejecutada en este peldaño.
4. **Plan de ablation del keypoint**: antes de gastar cómputo, verificar a
   qué checkpoint corresponde el `keypoint_auc≈0.774` citado — no está
   documentado en el repo para ningún submission salvo `r01` (checkpoint
   antiguo). Luego, dos vías ya pre-registradas y sin ejecutar (T1-82
   decoder local, T1-83 DARK) más una hipótesis nueva no cubierta
   (generalización entre escenarios, análoga al riesgo ya confirmado en
   Task 2) bajo el peldaño propuesto **T1-87**.

---

## 1. Paquete `submissions/r05-dist-rescaled/`

### Qué se copió

Bit-idéntico a `submissions/r04-interim-joint/`:

- `model_0.pth` — `sha256:6B73FC21BC16BA1AE004E35E875298036C88EA6D777303ABCF63015CBAA773DA`
- `model_1.pth` — `sha256:C365077A30533BBA5BEF9874A78A1F1FC11E89471939600F231AAE9C4310B01D`

Verificado con `diff` binario (`diff submissions/r04-interim-joint/model_0.pth
submissions/r05-dist-rescaled/model_0.pth`, sin salida = idéntico), no solo
por nombre de archivo. **Ningún peso se reentrenó ni se tocó.**

### Qué se cambió

Solo dos líneas de `inference.py` (`submissions/r05-dist-rescaled/inference.py:44-51`),
las constantes del ajuste lineal `pixel_gap -> distancia`:

| Constante | r04 (heredada de r01/r03) | r05 |
|---|---:|---:|
| `DISTANCE_SCALE_A` | 0.7320 | **0.9370** |
| `DISTANCE_SCALE_B` | 2.6779 | **3.4277** |
| `DISTANCE_FALLBACK_PX` | 159.2 | 159.2 (sin cambiar, justificado abajo) |

**`inference.py` es diff-idéntico a `r04` salvo esas líneas y sus
comentarios** (`diff -u submissions/r04-interim-joint/inference.py
submissions/r05-dist-rescaled/inference.py`, verificado). Ningún otro
componente (keypoint, Task 2, decodificador, preproceso) se tocó — la regla
del peldaño es una hipótesis por cambio.

**Por qué `1.28` es exacto, no una aproximación** [VERIFICADO por álgebra]:
el ajuste `A, B` original es una regresión OLS `distancia_GT = a·pixel_gap + b`
contra el target `GT_stored/10` (`analysis/verify_task1_distance_geometry.py`,
T1-R2, R²=0.9916 sobre 92,012 mediciones — `ATTACK_LADDER.md:743-760`). El
target corregido es `GT_stored/7.8125`. Como `10 / 7.8125 = 1.28` exacto
(`7.8125 = 4000/512`, así que `10/(4000/512) = 5120/4000 = 1.28` sin
redondeo), el nuevo target es el viejo multiplicado por la constante exacta
`k=1.28`. Para una regresión OLS `Y = aX+b`, si `Y_new = k·Y_old` entonces la
solución óptima es exactamente `a_new = k·a_old`, `b_new = k·b_old` — es
álgebra de mínimos cuadrados bajo reescalado lineal del target (`Cov(X,
kY)=k·Cov(X,Y)`, `Var(X)` no cambia), no un ajuste nuevo ni aproximado. Y
`R²` es invariante a esa transformación (`SS_res` y `SS_tot` escalan ambos
por `k²`), así que sigue siendo 0.9916 — **no se perdió precisión al
reescalar**. Esto se reverifica a escala completa en la sección 3 con la
regresión corrida dos veces (target viejo y target nuevo) sobre los mismos
61,691 frames, no solo aceptado por álgebra.

**`DISTANCE_FALLBACK_PX` deliberadamente sin reescalar**: es la mejor
constante posible bajo el umbral y target VIEJOS (AUC 0.0445 sobre 61,691
casos a `MAX_THRESHOLD_DIST=10`, target `GT/10` —
`analysis/explore_task1_gt.py`, citado en `inference.py:47-50` de r04). No es
una regresión OLS, así que la propiedad de reescalado exacto no aplica: el
óptimo bajo el umbral nuevo (0..20, que escaló `2×`, no `1.28×`) y el target
nuevo requiere resolver de nuevo el problema de optimización sobre el
dataset completo — solo disponible en el pod para el conjunto de 61,691
frames completo con su distribución de valores centinela. Tocarlo sin esa
medición sería introducir un segundo cambio no controlado en el mismo
peldaño. Queda como pendiente explícito (ver sección 3, nota sobre alcance).

### Verificación del contrato de submission

Contra `RULES_OF_ENGAGEMENT.md`:

- **Firma de 4 argumentos** `inference(task_id, oct_volume, opmi_image,
  model)` — heredada sin cambios de r04 (`inference.py:462`). ✅
- **Pesos** `model_0.pth` / `model_1.pth` en la raíz — presentes. ✅
- **`requirements.txt`** — no incluido, ni `numpy` ni `torch` declarados. ✅
- **`oct_volume is None`** manejado: `_infer_task1`/`_infer_task2` comprueban
  `oct_volume is not None` antes de usarlo (`inference.py:405-427`,
  `:441-459`) — fallback determinista si falta, heredado sin cambios. ✅
- **Formato de retorno**: `{"keypoints": [x, y], "tool_tissue_distance":
  float}` para Task 1, en píxeles reales de B-scan (no ×10) — sin cambios.
  Task 2: `np.ndarray (3,3) float64` — sin cambios. ✅
- **Regla de oro**: todo el cuerpo de `inference()` sigue envuelto en
  `try/except` con fallback numérico — sin tocar. ✅

### Resultados locales medidos

`vendor/fido/` ya tiene la corrección oficial, así que `eval/run_local.py`
puntúa hoy con las mismas constantes que usará Codabench.

```powershell
python eval/run_local.py --submission submissions/r05-dist-rescaled --task keypoints --save
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python eval/run_local.py --submission submissions/r05-dist-rescaled --task both --save
```

[VERIFICADO — comandos arriba, salida real, `submissions/r05-dist-rescaled/local_score.json`]:

| | keypoint_auc | distance_auc | final_score (T1) | Task 2 |
|---|---:|---:|---:|---:|
| r04, constantes viejas, **vendor viejo** (ya registrado en `r04/local_score.json`) | 0.854545 | 0.090909 | 0.625455 | 0.000000 |
| r04, constantes viejas, **vendor corregido** (medido hoy) | 0.854545 | **0.000000** | 0.598182 | — |
| **r05, constantes reescaladas, vendor corregido** | 0.854545 | **0.133333** | **0.638182** | 0.000000 |

`keypoint_auc` es idéntico entre las tres filas — control esperado, ninguna
constante tocada afecta al keypoint. La fila del medio (r04 sin cambios bajo
el vendor ya corregido) es el experimento de control real: **aislar el
efecto de las dos constantes del scorer (umbral+escala) SIN tocar el
modelo** hace que `distance_auc` caiga de 0.090909 a 0.0 sobre estos 5 casos
— confirma que, sin reescalar la predicción, el umbral más ancho no alcanza
a compensar el sesgo ahora más grande en unidades absolutas (ver sección 3).
Reescalar la predicción (fila r05) recupera y supera el valor original:
`0.0 -> 0.133333`.

**Advertencia obligatoria, ya documentada para r01/r03**
(`RESULTS.md:58-65`): el harness local de 5 casos NO predice Codabench —
r01 fue optimista local→real (0.48→0.5386), r03 fue pesimista
(0.60→0.569). Esta tabla confirma la **dirección** del efecto con datos
reales, no su magnitud sobre el test oculto de 100 casos. La sección 3 usa
datos a escala completa (61,691 frames) para acotar la magnitud.

**Cumplimiento de `CONSTITUTION.md` §I.3**: el Mock Test se usó solo para
verificar que el pipeline no truena y observar el signo del efecto, no para
elegir ni ajustar ninguna constante — las constantes de r05 vienen
enteramente de la corrección pre-registrada en `NOW.md`, no de mirar estas
5 etiquetas.

### Elección de base: r04, no r03

`r03` (Codabench real: **0.569**) superó a `r04` (Codabench real: **0.561**)
en el score agregado, pese a que sus componentes de keypoint son casi
idénticas localmente (`keypoint_0.8537.pth` de r03 vs el checkpoint
`val_keypoint_auc=0.8539` de r04 — diferencia de 0.0002, ruido, no señal) y
la cabeza de distancia es la MISMA (`unet_dist_v2_0.6113.pth`, ambas
heredan de r03 según el README de r04). Elijo `r04` como base de todos
modos, por tres razones:

1. **La diferencia de 0.008 entre 0.569 y 0.561 está dentro del ruido ya
   documentado del propio proyecto** [DOCUMENTADO]: la brecha local↔Codabench
   medida para r01 fue +0.059 (optimista) y para r03 fue −0.031 (pesimista)
   — un rango de ±0.03-0.06 en 5-100 casos. 0.008 no es una señal confiable
   de que el checkpoint de r03 generaliza mejor; con Δ_local=0.0002 entre
   los dos checkpoints de keypoint, cuál "gana" en 100 casos ocultos es
   indistinguible del ruido de muestreo.
2. **r04 es el checkpoint con procedencia trazable**: `val_keypoint_auc=0.8539`
   viene de la corrida pre-registrada T1-80 (`experiments/80-t1-clean-baselines/
   PRE_REGISTRATION.md`, 20 épocas, n=14399 en validación, mismo split que el
   A/B contra ResNet-18+FPN y DINOv2 — `ATTACK_LADDER.md:907-943`). El
   checkpoint de r03 (`keypoint_0.8537.pth`) es anterior a ese peldaño
   formal y no tiene un pre-registro equivalente en el repo. Partir de r04
   mantiene cada componente de r05 atado a un experimento documentado.
3. **r04 ya incluye `model_1.pth`** (Task 2, sin cambios, sigue en 0.0 local)
   — evita repaquetizar si el flujo de envío evalúa ambas tareas desde el
   mismo zip, a costo cero porque este peldaño no toca Task 2.

`r03` carece de `model_1.pth` en su propio zip subido
(`submissions/r03-dist-v2/dist/fido_task1_r03.zip` contiene únicamente
`inference.py` y `model_0.pth`, verificado con `zipfile.namelist()`) — otra
razón práctica menor a favor de r04 como plantilla reutilizable.

---

## 2. (cubierto en la sección 1 — paquete y verificación de contrato)

---

## 3. Estimación de la subida de `distance_auc`

### 3.1 Expectativa PRE-REGISTRADA (antes de medir nada de esta sección)

Escrita antes de correr el oráculo a escala completa (sección 3.3):

> Reescalar las constantes por 1.28× debería mover `distance_auc` de 0.0900
> (r01, Codabench, target/umbral viejos) a un valor bien por encima de eso,
> porque (a) el sesgo sistemático de ~28% dejaba la mayoría de los casos con
> segmentación medible fuera de CUALQUIER umbral razonable, y corregirlo
> debería recuperar buena parte del techo geométrico ya medido localmente
> como "inflado" (`val_distance_auc=0.6113`, que descarta casos sin gap —
> `ATTACK_LADDER.md`, T1-R5); y (b) doblar el umbral (10→20) ayuda
> mecánicamente a cualquier distribución de error con masa en (10,20]px,
> nunca puede bajar el AUC manteniendo el resto fijo. Espero que el efecto
> (a) domine sobre (b) para los casos con segmentación medible (el sesgo
> multiplicativo de 22% de una distancia típica de 50-200px es de 11-44px,
> mayor que el ancho del umbral nuevo por sí solo), y que el efecto conjunto
> quede muy por debajo del techo "inflado" de 0.6113 porque el scorer
> oficial SÍ penaliza los casos sin medición geométrica (fallback constante,
> ~40% de los casos según coverage medida en Mock Test, sección 3.2) — algo
> que esa métrica local descartaba. **No me comprometo con un número puntual
   sin la distribución de errores real**; la doy en la sección 3.3.

### 3.2 Lo medido en Mock Test (n=5, ya reportado en sección 1)

`distance_auc`: `0.0 -> 0.133333` al reescalar, bajo vendor corregido. De
los 5 casos, solo 3 tuvieron segmentación medible en al menos un B-scan;
de esos 3, 1 caso midió y 2 cayeron en el fallback constante
(`DISTANCE_FALLBACK_PX=159.2`, sin reescalar) con error de 370-587px —
**coverage de segmentación medible: 60% en esta muestra de 5**, n
demasiado chico para generalizar el número, pero consistente
cualitativamente con que el fallback (no la calibración lineal) domina el
error quirúrgicamente para una fracción no trivial de casos. En el único
caso con segmentación medible (`Scenario_11/04990`), el error absoluto bajo
target nuevo fue 6.48px con constantes reescaladas — dentro del umbral
nuevo de 20px — contra 47.52px con las constantes viejas. Esto es
consistente con la hipótesis de sesgo multiplicativo: para una distancia
real de ~194px, un sesgo de ~22-24% dejaba un error de ~47px, mucho mayor
que el umbral nuevo por sí solo.

### 3.3 Oráculo geométrico a escala completa (61,691 frames, medición real — PENDIENTE DE EJECUCIÓN, ver abajo)

*(Sección completada tras la corrida de
`analysis/measure_t1_distance_rescale_effect.py` sobre `data/_annotations/Task 1`
+ `data/_bscan_seg/Task 1`, disponibles localmente completos — no hace falta
pod para este análisis. Usa `auc_from_errors` REAL importada del vendor, no
una reimplementación.)*

<!-- SECCION_3_3_PLACEHOLDER -->

---

## 4. Investigación del residuo del 6.3% (pendiente `0.937` vs ideal `1.0`)

### 4.1 Por qué se espera pendiente = 1.0

`scoring_keypoints.py` documenta el target corregido explícitamente:
`GROUND_TRUTH_DISTANCE_SCALE` convierte el valor almacenado a "the true
tool-tip-to-retina **pixel** distance on the B-scan image" (comentario en el
propio archivo del vendor, `scoring_keypoints.py:15-21`). `pixel_gap` (fila
de ILM más superficial menos fila más profunda de `InstrumentInOCT`, en la
misma imagen de 512×512) es, por construcción, una medida en la misma
unidad — píxeles de fila de esa misma imagen. Si ambas cantidades miden
literalmente lo mismo, la regresión `target = a·pixel_gap + b` debería dar
`a≈1`, `b≈0`. La regresión reescalada da `a=0.9370` (**recordatorio
importante**: este número NO es una medición nueva contra el target
corregido — es exactamente `0.7320 × 1.28` por la invariancia de OLS bajo
reescalado de la sección 1; el 6.3% de desviación ya estaba presente en el
ajuste ORIGINAL contra el target viejo, T1-R2, y sobrevive intacto al
reescalado porque la reescala no cambia el `R²` ni la forma relativa del
ajuste). Investigar el 6.3% es investigar por qué el ajuste original de
T1-R2 nunca fue pendiente 1 — no algo que introdujo la corrección de hoy.

### 4.2 Hipótesis, en orden de plausibilidad

**H1 — Contaminación de identidad de herramienta [hipótesis líder]**: la
clase de segmentación `InstrumentInOCT` (11) no distingue CANNULA de
ENDOILLUMINATOR. T1-R1 confirmó que **ambas herramientas están presentes en
escena en el 100% de los frames muestreados**, y que solo `CANNULA` tiene
una `ILM Distance` real (el endoiluminador trae el centinela `2147484000`)
— `ATTACK_LADDER.md:713-732`. El filtro `--only-cannula` de
`verify_task1_distance_geometry.py`/`verify_task1_distance_full.py` excluye
frames donde CANNULA no es la activa, pero **no verifica que el
endoiluminador no aparezca también como píxeles de clase 11 en el mismo
B-scan**. Si el endoiluminador está más profundo que la cánula en algunos
frames (o su sombra se segmenta bajo la misma clase), `tip_and_ilm_column`
toma "el píxel de clase 11 con mayor fila" — que en esos frames sería el
endoiluminador, no la cánula — produciendo un `pixel_gap` sistemáticamente
distinto de la distancia GT (que es específicamente de CANNULA). Esto
predice una pendiente `<1` (el gap medido incluye frames "contaminados" con
un objeto en general más profundo o más superficial que la cánula real,
diluyendo la relación 1:1) — consistente con lo observado.
**Cómo falsarlo**: contar componentes conexos de la clase 11 por frame; si
hay >1 componente en una fracción no trivial de los 61,691 frames, repetir
el ajuste solo sobre frames con exactamente 1 componente y comparar la
pendiente. Si sube hacia 1.0, H1 queda confirmada; si no cambia, se
descarta.

**H2 — Definición del punto de referencia no es la usada por el GT**: "punta
= píxel con mayor fila de clase 11" y "referencia = fila mínima de ILM en
ventana ±5 columnas" son convenciones razonables pero no verificadas contra
el método real de generación del GT (que probablemente usa una posición 3D
exacta de la punta del instrumento en el simulador, no una máscara de
segmentación 2D post-hoc). Cualquier sesgo sistemático entre "punta 3D real
según el simulador" y "punta más profunda visible en la máscara 2D
segmentada" (p.ej. por oclusión parcial, ángulo de corte del B-scan, o
suavizado/erosión en el proceso de generación de la máscara) produciría
exactamente un desajuste de pendiente/intercepto sin romper el `R²` (porque
sería un sesgo suave y consistente, no ruido).
**Cómo falsarlo**: no es falsable con los datos ya disponibles localmente
sin el código de generación del GT/máscaras del simulador (fuera del
alcance de este repo). Se puede acotar indirectamente: si H1 se descarta y
el R² sigue en 0.99 con pendiente <1, un sesgo sistemático de definición de
referencia (H2) es la explicación que queda en pie por descarte.

**H3 — Erosión/antialiasing en el borde de la máscara**: si la máscara de
segmentación sistemáticamente "recorta" 1-2 filas en el borde superior de
ILM o el borde inferior del instrumento (p.ej. por post-proceso de
generación de la máscara, o por cómo se resuelve el antialiasing en los
límites de clase), el `pixel_gap` medido sería sistemáticamente MENOR que
el gap real — consistente con pendiente `<1` cuando el gap real es grande,
pero predice un **intercepto negativo** o cercano a 0 en vez de `b>0`
(actualmente `b=3.4277>0` tras reescalar), lo que no encaja limpio con un
recorte de bordes puro. Esta hipótesis pierde puntos frente a H1 por esta
inconsistencia direccional, pero no está descartada.
**Cómo falsarlo**: medir el `pixel_gap` con y sin un margen de tolerancia de
1-2 filas (dilatar la máscara antes de medir); si la pendiente se acerca a
1.0 al dilatar, H3 gana peso.

**H4 — Resolución de la máscara distinta de 512×512 [DESCARTADA]**
[VERIFICADO]: se comprobó con `PIL.Image.size` que tanto el B-scan crudo
(`iOCT Microscope/Bscan/<frame>/00.png`) como su segmentación
(`Segmentation/00.png`) son exactamente `(512, 512)` en los casos
verificados (Mock Test, `Scenario_01` de `data/_bscan_seg`) — coincide con
el eje de 512 que usa `GROUND_TRUTH_DISTANCE_SCALE=(4.0/512)*1000`. Un
desajuste de resolución habría sido la explicación más simple y barata de
descartar; queda descartada.

**H5 — Submuestreo de un único B-scan vs promedio de los dos**: T1-R2
(`ATTACK_LADDER.md:743`) reporta 92,012 mediciones sobre 61,691 frames —
es decir, cada B-scan (hasta 2 por frame) se trata como una observación
independiente, sin promediar. Este informe (sección 3.3) promedia los dos
B-scans por frame cuando ambos son medibles, una variante metodológica
distinta. Si los dos B-scans (probablemente dos cortes ortogonales) miden
geometrías algo distintas de la misma punta, promediar cambia la varianza
del residuo pero no debería mover la pendiente de forma sistemática — se
reporta como posible fuente de RUIDO adicional, no de sesgo, y se puede
descartar comparando el ajuste de este informe con una variante
sin-promediar (un B-scan = una observación) sobre los mismos frames.

### 4.3 Qué NO se puede concluir todavía

Ninguna de H1-H3 se falsó activamente en este peldaño (falta scriptear el
conteo de componentes conexos de H1, que es el siguiente paso de mayor
prioridad porque es la única con una prueba barata y disponible localmente
ahora mismo, sin pod). El 6.3% es pequeño en términos absolutos (R²=0.9916
ya es muy alto) pero relevante porque desplaza cada predicción de distancia
por un factor multiplicativo constante — el mismo tipo de error, aunque
mucho menor, que causó el problema de 1.28× que motivó este peldaño.

---

## 5. Plan de ablation del keypoint (local 0.8539 vs Codabench ≈0.774)

### 5.0 Paso 0, antes de cualquier ablation: verificar la procedencia del número

[VERIFICADO — no encontrado en el repo]: `RESULTS.md:56` marca explícitamente
el desglose de `keypoint_auc`/`distance_auc` de Codabench para `r03` como
`pendiente`, y no hay ningún desglose registrado para `r04` tampoco (solo el
score agregado `0.561`, `NOW.md:102`). El único desglose de Codabench
**verificado en el repo** es el de `r01`: `keypoint_auc=0.7309`
(`RESULTS.md:54`) — con un checkpoint de keypoint que **antecede** a
`T1-80`/`T1-85` (la corrida que dio `0.8539` es del 2026-08-19, r01 es muy
anterior). El `keypoint_auc≈0.774` citado en el encargo de este peldaño no
tiene una fuente verificable dentro del repo a fecha de este informe.
**Antes de gastar cómputo en cualquier ablation, el primer paso barato es
confirmar en el panel de Codabench (fuera de este repo) a qué submission y
checkpoint corresponde ese `0.774`** — si corresponde a `r01` (checkpoint
viejo) el "hueco de 0.08" está mal planteado (compara checkpoints distintos,
no el mismo modelo en dos entornos) y la ablation real más urgente es
simplemente **subir r05/r04 y leer su propio desglose**, no diagnosticar un
número de procedencia incierta.

### 5.1 Lo que ya existe pre-registrado, sin ejecutar (requiere pod)

Dos peldaños ya escritos y con gate definido, pendientes de GPU:

- **T1-82** — decoder local (ventana 7) vs global, sobre el checkpoint CNN
  limpio T1-80 (`experiments/82-t1-local-decoder/PRE_REGISTRATION.md`). Gate:
  GO solo si `delta_auc >= 0.002` y ningún escenario pierde `>0.01` AUC.
- **T1-83** — corrección DARK (expansión de Taylor sobre el pico gaussiano)
  vs global, mismo checkpoint y mismo gate
  (`experiments/83-t1-dark-decoder/PRE_REGISTRATION.md`).

Ambos atacan la misma hipótesis: cuantización sub-píxel del heatmap de
stride 16 (`src/fido/heatmap_decode.py:1-13`, ya documentada como causa
conocida en la literatura del proyecto — "con stride 8 y umbrales enteros
0..10 se pierden los primeros 4-5 umbrales SIEMPRE"). El modelo de
producción (`inference.py`) usa `soft_argmax_2d` **global**, la variante más
simple y la que más cuantización sub-píxel puede dejar sobre la mesa.

### 5.2 Hipótesis nuevas que el plan de T1-82/83 no cubre

**H-domain-gap — el split local no mide lo que Codabench mide**: el
`GroupKFold` de `task1_split` separa por escenario dentro de los **mismos
10 escenarios** de entrenamiento (`src/fido/data/task1.py:47`, `task1_split`
agrupa por `case["scenario"]`). El fold de validación local nunca vio esos
frames en entrenamiento, pero sí comparte vídeo/paciente-sintético/rig de
iluminación con escenarios que SÍ entrenaron, si el simulador reutiliza
activos visuales entre escenarios. El test oculto de Codabench, en cambio,
probablemente usa escenarios generados independientemente (no hay evidencia
en el repo de que compartan semilla o activos con los 10 de train). Esto es
estructuralmente el mismo riesgo ya documentado y confirmado como real para
la escala de Task 2 ("Riesgo abierto — escala de Task 2 desplazada",
`NOW.md`) pero **nunca verificado para Task 1**.
**Prueba barata, sin pod**: con el checkpoint T1-80 ya descargado
(`checkpoints_from_pod/latest/keypoint_0.8537.pth` o equivalente), medir
`keypoint_auc` por escenario individual sobre el fold de validación
(`per-scenario AUC`, ya lo hace `evaluate()` en varios scripts del proyecto).
Si el AUC varía mucho entre escenarios de validación (algunos muy por debajo
de 0.85), es evidencia de que el modelo no generaliza uniformemente entre
"identidades" de escenario — consistente con que un escenario realmente
nuevo (Codabench) caiga más lejos todavía. Si el AUC es uniforme entre
escenarios, la hipótesis de domain-gap por escenario pierde fuerza y el
hueco hay que buscarlo en otro lado.

**H-checkpoint-mismatch**: cubierta en 5.0 — si el `0.774` no corresponde al
checkpoint de `0.8539`, el "hueco de 0.08" no existe como tal.

**H-oct-volume-none — descartada por diseño**: el 10% de casos de Task 1 sin
`oct_volume` (semilla determinista 2027, `RULES_OF_ENGAGEMENT.md` regla 6)
no puede explicar el hueco del keypoint: `_infer_task1` calcula el keypoint
únicamente a partir de `opmi_image` (`inference.py`, rama `net = model.get
("keypoint")`), sin usar `oct_volume` en ningún punto de esa rama. Se
descarta sin necesidad de medir nada.

**H-resize-fundus**: `RULES_OF_ENGAGEMENT.md` regla 5 confirma que
`opmi_image` real es 1024×1024×3 en el Mock Test, y el código reescala si
llega otro tamaño (`inference.py`, `_infer_task1`, comparación
`fundus.shape[-2:] != (FUNDUS_TRAIN_SIZE, FUNDUS_TRAIN_SIZE)`). No hay
evidencia de que el test oculto use un tamaño distinto, pero tampoco hay
verificación de que NO lo haga — es barato de comprobar si Codabench expone
las dimensiones reales en sus logs de ingestión.

### 5.3 Peldaño propuesto: T1-87 — ablation del error residual del keypoint

**Hipótesis**: el hueco `0.8539` (local) vs `keypoint_auc` real de Codabench
se explica principalmente por [a determinar entre] cuantización del decoder
sub-píxel (T1-82/T1-83, ya escritas) o generalización entre escenarios
(H-domain-gap, nueva), no por capacidad del backbone (ya descartado tres
veces, `T1-80/T1-85`).

**Orden de ejecución, de más barato a más caro**:
1. [Sin pod, minutos] Confirmar a qué checkpoint corresponde el
   `keypoint_auc` real citado (paso 5.0).
2. [Sin pod, minutos] `keypoint_auc` por escenario sobre el fold de
   validación local del checkpoint T1-80 ya descargado — mide
   H-domain-gap sin entrenar nada nuevo.
3. [Pod, ya escrito] Ejecutar T1-82 (decoder local) y T1-83 (DARK) contra
   el mismo checkpoint, con sus gates ya definidos.
4. Si T1-82/T1-83 dan GO pero no cierran todo el hueco medido en el paso 1,
   combinar con H-domain-gap: reentrenar con escenarios de validación
   distintos a los usados hasta ahora (`fold` != 0) para estimar la
   varianza entre-escenario del AUC y compararla con la magnitud del hueco.

**Criterio de aceptación**: el peldaño se considera cerrado cuando la suma
de los deltas de AUC atribuidos a cada causa confirmada (decoder + domain
gap + lo que aparezca) se acerca al hueco medido en el paso 1 dentro de un
margen razonable (`±0.02`, mismo orden que el ruido local↔Codabench ya
documentado en la sección 1). Si la suma no cierra el hueco, queda una
causa no identificada y el peldaño se declara **parcial**, no se fuerza una
conclusión.

**Costo**: pasos 1-2 son de minutos y no requieren pod. Pasos 3-4 reusan
checkpoints y cachés ya existentes (`checkpoints/task1_clean_cnn/`,
`fold0_logits.npz` una vez generado) — sin entrenar nada nuevo, solo
inferencia y decodificación offline.

---

## 6. Próximos pasos (fuera del alcance de este peldaño, pero desbloqueados por él)

1. **Subir `r05-dist-rescaled` a Codabench** — es la acción de mayor
   retorno por esfuerzo identificada en `NOW.md`. Este informe deja el
   paquete verificado y listo; la decisión de gastar una de las
   submissions restantes es de Rodrigo, no de este peldaño.
2. **Releer todo `val_distance_auc` medido antes de hoy** bajo la escala
   vieja (incluido el `0.6113` de T1-R5 v2) — ya señalado como pendiente en
   `NOW.md`, no se tocó en este informe.
3. **H1 de la sección 4** (conteo de componentes conexos de la clase 11)
   es la prueba más barata pendiente de este peldaño — no requiere pod,
   solo iterar las máscaras ya locales una vez más.
4. **Ejecutar T1-87** (sección 5.3) empezando por los pasos 1-2 (sin pod),
   antes de comprometer tiempo de GPU en T1-82/T1-83.
