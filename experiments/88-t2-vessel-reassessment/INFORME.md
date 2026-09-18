# T2-88 — Reevaluación crítica de la vía de vasos a la luz de MEMO/VDD-Reg y STHN/UASTHN

**Fecha**: 2026-08-19. **Mandato**: auditar la conclusión "vía de vasos muerta"
(`experiments/86-t2-postmortem/INFORME.md`, que a su vez se apoya en T2-R4,
T2-83 y T2-82) contra dos papers no pesados antes: MEMO/VDD-Reg (registración
retiniana por vasos) y STHN/UASTHN (geolocalización térmico-satelital de
template pequeño). No se escribe código de producción ni se relitiga nada
salvo lo que la tarea pide explícitamente re-auditar.

**Convención de lectura**: cada afirmación de código lleva `archivo:línea`.
Cada afirmación de un paper lleva sección y página del PDF (extraído con
`pdftotext -layout` y verificado con `PyMuPDF` para la numeración de página).
Donde razono en vez de medir, digo "inferencia" explícitamente.

---

## Veredicto (adelantado)

**La vía de vasos sigue muerta, y con una base más sólida de lo que sugería
el marco de "solo se probó apariencia".** El oráculo que la mató (T2-83 +
T2-R4) **no midió apariencia/intensidad — midió estructura**: segmentación
de vaso contra segmentación de vaso, y keypoint de vaso contra máscara de
vaso, en la pose GT exacta, validado contra un control positivo real
(instrumento) dentro del mismo arnés. MEMO ataca un problema distinto
(densidad de vaso diferente **del mismo árbol vascular físico**, visible con
distinta sensibilidad en cada modalidad); ese problema no es el que tiene
FIDO, donde el oráculo GT-contra-GT ya muestra que no hay correspondencia
por encima del azar en ninguna de las 8 orientaciones posibles. STHN/UASTHN
no aportan una vía nueva de señal — son técnicas de búsqueda gruesa-a-fina y
de estimación de incertidumbre que **presuponen** que existe una
representación con correlación cross-modal aprendible; FIDO ya probó esa
representación (T2-82, CNN completamente entrenable) y colapsó. Además, el
régimen de tamaño de FIDO (2.4% de área) es ~4.6x más extremo que el caso
más duro que STHN reporta haber evaluado (11% de área, no 11% de lado — ver
§3). Ningún elemento de los dos papers invalida la refutación existente.

---

## 1. ¿Qué midió realmente el oráculo de T2-83/T2-R4: apariencia o estructura?

### 1.1 T2-83 — keypoint de vaso del fundus contra máscara de segmentación OCT

`analysis/verify_task2_enface_convention.py:105` construye `vessel_uv` a
partir de `points_from_group(data, "Vasculature")` — **puntos de bifurcación
vascular anotados por el organizador sobre el fundus** (verificado leyendo un
caso real: `data/Mock Test/Task 2/Scenario_12/Numerical/00031.json` tiene 56
claves `V_0..V_55` con coordenadas en píxeles de fundus, incluyendo valores
negativos, es decir cubren todo el árbol visible, no solo la huella OCT). La
línea 114 compara esos puntos, proyectados a `uv` con la matriz GT
(`map_fundus_to_uv`, línea 37-40), contra `np.any(seg == VESSEL_CLASS,
axis=1)` — la **máscara de segmentación real del volumen OCT** (clase 3,
`ArteriesOrVeins`, confirmada en T2-R3, `ATTACK_LADDER.md:318-333`), colapsada
sobre el eje de profundidad. La métrica es distancia del punto a la máscara
binaria vía `distance_transform_edt` (`verify_task2_enface_convention.py:74-77`),
no correlación de intensidad. **Esto es una comparación estructura-contra-estructura
(keypoint anotado vs. máscara segmentada), no apariencia.**

`analysis/validate_task2_enface_convention_train.py` reutiliza exactamente
esa función (`measure_case`, importada en la línea 19-21) sobre el fold de
train real (no Mock), 966 casos, con la convención en-face ya corregida por
T2-83 (`transpose__flip_u__flip_v`). Resultado real
(`experiments/83-t2-enface-convention/oracle_train.md`, tabla "Vasculature",
n=906 puntos pooled sobre las 8 orientaciones): ratios observado/null en
**[0.8116, 1.1033]** en los cuatro radios (0.005 a 0.04, fracción del lado)
para las 8 variantes de convención — planos alrededor de 1.0, sin
enriquecimiento en ninguna orientación.

**Control positivo, en el mismo arnés, misma corrida**: la tabla "Instrument"
del mismo archivo da, para `transpose__flip_u__flip_v` (la convención
correcta), ratios **3.44 / 3.40 / 3.60 / 3.01** en los cuatro radios — y
ninguna otra de las 8 variantes se acerca. Esto es decisivo: **el mismo
arnés, con el mismo tipo de métrica (punto vs. máscara segmentada,
distance-transform), sí detecta señal estructural fuerte cuando existe**
(instrumento). Su ausencia total para vasos no puede explicarse por un bug de
convención o de ejes — ya está descartado por el propio control positivo — y
tampoco por baja potencia estadística: a radio pequeño (r=0.005, tasa null de
solo 8.5%, no saturada) el ratio vascular sigue en ~0.91-1.11, no hay
enriquecimiento ni siquiera donde el problema de base-rate alto (a r=0.04 la
máscara cubre ~47% del área, ver tabla) no aplica.

### 1.2 T2-R4 — proyección en-face de vasos contra máscara de segmentación del fundus (NCC)

`analysis/verify_vessel_signal_candidates.py:102-103` carga
`fundus_vessel_gt = (load_label_map(vessel_mask_path) > 0)` desde
`Stereo Left/.../Segmentation/arteriesorveins.png` — **la máscara de
segmentación real del lado fundus**, no la imagen RGB cruda. La compara
(`oracle_ncc_for_case`, líneas 96-152) contra distintas derivaciones de la
señal OCT (`enface_vessel_density`, binarizada, binarizada+dilatada,
`sig_binary` en la línea 84-86) usando la pose GT exacta
(`decompose_similarity`, sin ninguna búsqueda). Es, de nuevo,
**segmentación-contra-segmentación**, no intensidad-contra-intensidad.
Resultado (`ATTACK_LADDER.md:371-401`, `analysis/verify_vessel_signal_candidates.py:1-56`):
NCC medio en **[0.0071, 0.0181]** para las variantes A-D, y **[0.001, 0.010]**
con dilatación simétrica de hasta 21px por lado (variante E) — muy por debajo
del umbral orientativo 0.3. La verificación F (búsqueda local con traslación
libre) encuentra NCC≈0.24-0.31 pero en una posición que **no coincide con el
GT** en ninguno de los 4 casos — confirmando que el árbol vascular es
autosimilar (cualquier parche de curvas finas se parece a cualquier otro), el
mismo modo de fallo que la literatura de geolocalización (§3) documenta para
terrenos de baja textura.

### 1.3 Lo que sí midió solo apariencia — y por qué no es la evidencia relevante

`T2-82` (`src/fido/train/train_task2_common.py:411`) instancia
`Task2Dataset(..., include_vessel_enface=False, ...)`, y
`src/fido/train/train_task2_baseline.py:156-162` hace lo mismo con un
comentario explícito: *"este baseline (T2-R2) no usa el campo enface_vessel
(motor de vasos T2-R4 refutado, ver ATTACK_LADDER.md)"*. El `OctDenseEncoder`
(`src/fido/models/task2_common.py:41-43`) toma **1 canal**, que es
`enface_projection` (media de intensidad sobre profundidad,
`src/fido/data/common.py:102-106`), no `enface_vessel_density`
(`src/fido/data/common.py:147-150`, función que existe pero está desconectada
de este pipeline por diseño). **T2-82 es, correctamente, un test de
apariencia/intensidad — nunca tocó el canal de vasos.** El postmortem T2-86
lo usa como evidencia de que "ninguna capacidad de red arregla esto" en un
sentido general (una CNN totalmente entrenable tampoco encuentra nada), lo
cual es válido como argumento sobre la señal de intensidad — pero **no es,
ni pretende ser, una prueba adicional sobre vasos**; esa prueba ya está
cubierta, y de forma más directa, por 1.1 y 1.2.

### Respuesta a la pregunta 1

**No, el oráculo de T2-83/T2-R4 no midió apariencia — midió estructura**, con
dos diseños independientes (keypoint-vs-máscara y máscara-vs-máscara), en la
pose GT exacta, y con un control positivo dentro del mismo arnés que confirma
que el método sí detecta señal cuando existe. La distinción que motivaba
releer MEMO ("si el oráculo midió apariencia, MEMO no está refutado") **no se
sostiene**: MEMO ataca exactamente el escenario de estructura-vs-estructura
que ya se probó y dio negativo.

---

## 2. MEMO / VDD-Reg: qué hace y si es aplicable

### 2.1 Qué problema resuelve MEMO (verificado, p.1 Abstract, p.1-2 §1)

MEMO registra pares **EMA-OCTA** de la misma retina de primate no humano. El
reto que motiva el paper es que la **densidad de vaso (VD)** difiere >30%
entre modalidades porque **EMA no puede ver los capilares más pequeños que sí
ve OCTA** ("most capillaries cannot be visualized in EMA images", Fig. 1,
p.1) — es decir, **ambas modalidades imagean el mismo árbol capilar físico
en el mismo plano retiniano**; la diferencia es de **sensibilidad/visibilidad
parcial**, no de correspondencia geométrica. Esta es la premisa que hace
funcionar a VDD-Reg: existe una estructura común real, solo que una modalidad
ve un subconjunto de ella.

### 2.2 Arquitectura VDD-Reg (verificado, §4, p.7-13)

1. **Módulo de segmentación (LVD-Seg, §4.1, p.7-9)**: dos etapas
   semi-supervisadas. Etapa 1 — MSE supervisado con **tan solo 3 máscaras
   anotadas** de EMA (la modalidad de menor densidad) para estabilizar el
   entrenamiento. Etapa 2 — *style loss* no supervisado que empuja al
   segmentador a extraer únicamente los vasos **visibles en ambas
   modalidades**, no todos los vasos reales. Salida: mapas de probabilidad
   por píxel (no binarios) para EMA y OCTA.
2. **Módulo de registración (§4.2, p.11-13)**: SuperPoint preentrenado
   (detector+descriptor) sobre los mapas de probabilidad, emparejamiento
   bidireccional del vecino más cercano, RANSAC para eliminar outliers, y
   ajuste de una **transformación afín parcial de 4 grados de libertad**
   (p.13, línea "the partial affine transformation (i.e., 4 degrees of
   freedom) was adopted") — el mismo número de DOF que la similitud
   reflejada de Task 2.
3. **Resultado en el propio dataset MEMO** (Tabla 4, p.16): 86.67% de éxito
   (RMSE<10px) frente a 60% del mejor baseline (CycleGAN-based) y 0-33% de
   los métodos restantes, sobre un test set de **30 pares totales**.

### 2.3 ¿Es aplicable a FIDO?

**Lo que FIDO ya tiene y MEMO necesitó resolver con ingeniería**: FIDO regala
máscara de vaso GT completa en ambos lados para prácticamente todo el
dataset de entrenamiento (fundus: `arteriesorveins.png`, confirmado por
T2-R3 y usado en §1.2; OCT: clase 3 `ArteriesOrVeins`, confirmado por T2-R3).
El motivo de ser de LVD-Seg — entrenar un segmentador razonable con solo 3
máscaras anotadas — **no aplica**: no hay escasez de etiquetas que resolver
en FIDO, así que ese componente entero de VDD-Reg se podría saltar
completamente y usar la máscara GT como entrada directa al módulo de
registración (SuperPoint + RANSAC 4-DOF).

**Lo que MEMO no resuelve para FIDO**: su método completo — incluyendo la
etapa 2 semi-supervisada, diseñada específicamente para "extraer solo los
vasos visibles en ambas modalidades" — asume que existe un subconjunto común
real que aislar. Esa asunción es exactamente la que el oráculo GT-contra-GT
de §1.1/§1.2 falsa para FIDO: usando la máscara GT **completa y perfecta**
de ambos lados (el mejor caso posible, sin ruido de segmentador), la
correspondencia en la pose GT exacta es indistinguible del azar. Si dos
fuentes de etiquetas perfectas no correlacionan geométricamente entre sí en
absoluto, no hay "subconjunto visible en ambas modalidades" que un
detector de puntos (SuperPoint u otro) o un `style loss` puedan aislar —
VDD-Reg mejora la robustez a densidad de vaso **diferente pero relacionada**;
no puede inventar relación donde no la hay en las etiquetas de origen.

**Qué habría que implementar si se quisiera probarlo de todos modos** (para
que la respuesta sea completa, no solo negativa): (a) generar mapas de
"vesselness" en ambos lados — trivial, ya existen las máscaras; (b) correr
SuperPoint (checkpoint público) sobre ambos mapas; (c) emparejar
bidireccionalmente + RANSAC para el afín 4-DOF; (d) medir tasa de éxito
RMSE<10px con la GT real de Task 2. Coste: bajo (CPU/GPU modesta, sin
entrenar nada si se usa SuperPoint congelado, quizás 1-2 días de ingeniería).
Pero es exactamente el "Motor 2 — registro aprendido" que
`ATTACK_LADDER.md:339-342` deja plantado a la espera de resolver el hallazgo
de `enface_vessel_density` — hallazgo que **sí se resolvió, en negativo**
(§1.2). Ejecutar esto ahora repetiría, con un detector de puntos más
sofisticado (SuperPoint en vez de NCC), un experimento cuyo insumo crítico
—la existencia de correspondencia geométrica en las etiquetas GT— ya se
midió ausente por partida doble.

---

## 3. STHN / UASTHN: ¿es de verdad comparable el régimen de tamaño?

### 3.1 Definición exacta de "size ratio" en STHN (verificado, §IV-A, p.4)

STHN fija el lado de la imagen térmica en `W_T = 512` px y varía el lado del
mapa satelital `W_S ∈ {512, 1024, 1536}`. El texto dice literalmente: *"For
W_S = 512/1024/1536, the size ratios between thermal images and satellite
images are 100%, 25%, and 11%"* (p.4). Verificación aritmética: `(512/512)^2
= 100%`, `(512/1024)^2 = 25%`, `(512/1536)^2 = 11.1%` — **esos porcentajes
son ratio de ÁREA, `(W_T/W_S)^2`, no ratio de lado**. El ratio de LADO en el
caso más duro de STHN es `512/1536 ≈ 33.3%`, no 11%.

**Esto corrige una premisa de la tarea**: la comparación "STHN 11% vs. FIDO
15.6% de lado / 2.4% de área" mezcla una cifra de área de STHN con una de
lado de FIDO. Comparando la misma unidad (área): **FIDO (2.4%) es ~4.6x más
extremo que el caso más duro que STHN evalúa (11%)**. En términos de lado:
FIDO (15.6%) es más pequeño que incluso el caso más fácil citado en abstracto
de STHN si se leyera mal como lado, pero correctamente comparado (33.3% de
lado en STHN vs. 15.6% en FIDO) FIDO sigue siendo sustancialmente más
extremo — el template ocupa una fracción lineal casi la mitad de pequeña.
STHN no reporta haber probado ratios de área más chicos que 11%; no hay
evidencia en el paper de que su método escale a un régimen tan extremo como
el de FIDO.

### 3.2 ¿Qué asume STHN que FIDO no cumple? (verificado, §II, §III, p.1-3)

El método (`F_H`, Ecs. 1-2, p.3) es un extractor de features CNN + volumen de
correlación + estimador iterativo de desplazamiento de 4 esquinas — la misma
familia arquitectónica (IHN) que ya está listada como referencia en
`ATTACK_LADDER.md` y conceptualmente equivalente a lo que T2-82 ya probó:
**una red totalmente entrenable que aprende una representación común y
calcula un volumen de correlación entre ambas modalidades**. STHN no aporta
un mecanismo distinto para *descubrir* correspondencia — su contribución es
*eficiencia de búsqueda* (evitar el muestreo denso del mapa completo,
Fig./§I p.1-2) y una etapa de refinamiento, ambas construidas **sobre** una
representación que ya correlaciona razonablemente bien (thermal/satélite
imagean literalmente el mismo terreno — carreteras, edificios, campos —
solo que con distinta modalidad de sensor). Esa es la asunción que FIDO no
cumple para vasos: T2-82, la versión sin restricción de capacidad de ese
mismo tipo de arquitectura de correlación, colapsó sin usar el OCT en
absoluto (`train_run.log`, `shuffle_drop_points` oscilando alrededor de 0
en las 10 épocas — ver también T2-86 §b). Construir la etapa "coarse-to-fine"
de STHN encima de una representación que ya se demostró sin señal no
resuelve el problema de fondo; sería optimizar la búsqueda de algo que no
está.

### 3.3 UASTHN: no crea señal, la caracteriza (verificado, Abstract y Fig.1, p.1)

UASTHN (p.1) es explícitamente una capa de **estimación de incertidumbre**
(`CropTTA` + Deep Ensembles) sobre un modelo DHE ya entrenado — no propone un
mecanismo nuevo de correspondencia. Es notable, y relevante en la otra
dirección (a favor de la solidez del hallazgo de FIDO, no en contra), que
UASTHN liste **"Self-similar Maps ... leading to false matches"** (Fig. 1(d),
p.1-2) como una de las seis categorías canónicas de fallo de alta
incertidumbre en este tipo de tarea — exactamente el mecanismo de fallo que
`ATTACK_LADDER.md:388-401` documentó de forma independiente para el árbol
vascular de FIDO (búsqueda local con traslación libre encuentra NCC alto en
una posición equivocada). La literatura de geolocalización coincide con el
hallazgo de FIDO en que la autosimilitud es un modo de fallo real y conocido,
no evidencia de que exista señal recuperable con más sofisticación.

### Respuesta a la pregunta 3

El régimen de tamaño de STHN, medido correctamente en área, es
**considerablemente menos extremo** que el de FIDO (11% vs. 2.4%), no
comparable. Su enfoque coarse-to-fine es arquitectónicamente afín a lo que
FIDO ya planea (`docs/superpowers/specs/2026-08-18-fido-task2-cross-modal-design.md:41-52`)
pero **asume una representación con correlación cross-modal aprendible que
FIDO ya probó (T2-82) y no encontró**, y ambos papers (STHN, UASTHN)
reconocen la autosimilitud como un modo de fallo no resuelto, coincidiendo
con — y no refutando — el hallazgo propio de FIDO.

---

## 4. Veredicto final

### La vía de vasos está muerta — confirmado, sobre bases más fuertes que las citadas originalmente

Las tres evidencias que el análisis previo cita (oráculo de apariencia
1214 casos, T2-83, T2-82) sostienen la conclusión, pero **la razón correcta
no es "solo se probó apariencia y falta probar estructura"** — la
estructura ya se probó, dos veces, de forma independiente (§1.1, §1.2), con
un control positivo dentro del mismo arnés que confirma que el método
detecta señal real cuando existe (instrumento, ratio 3.0-3.6x). Esa es una
refutación más fuerte de lo que el resumen "vías geométricas dependen de la
misma señal de apariencia" (`experiments/86-t2-postmortem/INFORME.md:257-269`)
sugiere: no dependen de una señal de *apariencia* ausente, dependen de una
*correspondencia geométrica* entre las máscaras GT que tampoco existe.

- **MEMO/VDD-Reg no rescata la vía** porque su método está diseñado para un
  problema distinto (mismo árbol vascular físico, visibilidad parcial
  diferente por sensibilidad de modalidad) que el que tiene FIDO
  (correspondencia geométrica ausente incluso con máscaras GT perfectas en
  ambos lados). El componente de VDD-Reg que sí sería nuevo para FIDO — el
  detector SuperPoint + RANSAC 4-DOF sobre mapas de vesselness — es
  exactamente el "Motor 2" que `ATTACK_LADDER.md` dejó pendiente de un
  hallazgo que ya se cerró en negativo.
- **STHN/UASTHN no rescatan la vía** porque atacan un problema de
  *eficiencia de búsqueda* y de *caracterización de incertidumbre* sobre una
  representación que ya correlaciona razonablemente (mismo terreno,
  distinto sensor); FIDO no tiene esa representación correlacionada para
  vasos, y el régimen de tamaño de FIDO es más extremo que el que estos
  papers evalúan.

### No se propone un peldaño nuevo de vasos

Dado que el hallazgo central (ausencia de correspondencia geométrica en las
etiquetas GT, no solo en la apariencia) es más fuerte de lo que se pensaba,
no hay una hipótesis pre-registrable razonable que un tercer intento de
vasos (con MEMO, con STHN, o con cualquier otra arquitectura) pudiera testear
de forma distinta a lo ya hecho: cualquier método, por sofisticado que sea,
opera sobre la misma información de entrada (las máscaras GT de vaso en
ambos lados) cuya correspondencia geométrica ya se midió indistinguible del
azar en la pose conocida correcta. **Gastar presupuesto en implementar
VDD-Reg o un clon de STHN sobre vasos tendría, con la evidencia actual,
retorno esperado ≈0** — no es un juicio de que el código fuera a fallar por
un bug, es que el insumo que ambos métodos necesitan (correlación real entre
las dos fuentes de etiquetas) ya se midió ausente con dos diseños de oráculo
independientes y una potencia estadística razonable (n=906-1214 puntos/casos).

**Recomendación**: se sostiene la recomendación de `experiments/86-t2-postmortem/INFORME.md`
(§c): el único peldaño de Task 2 con retorno esperado no nulo y aún no
ejecutado es T2-R12 (correlación no-visual entre el keypoint de Task 1 y la
traslación de Task 2), que no depende de ninguna señal de imagen —ni de
apariencia ni de estructura de vaso— y por tanto no está tocado por esta
auditoría ni por la refutación aquí confirmada.

### Lo que dejaría esto en duda (para que quede honesto, no cerrado con exceso de confianza)

- **No verificado en esta auditoría, solo inferido**: por qué, mecánicamente,
  el "ArteriesOrVeins" del volumen OCT colapsado en profundidad no
  corresponde al árbol vascular superficial del fundus renderizado. Es
  plausible —no medido aquí— que el simulador de FIDO genere la textura de
  vasos del fundus de forma independiente de la malla 3D usada para el
  volumen OCT (un problema de autoría de datos sintéticos, no de algoritmo).
  Si ese fuera el caso, ningún método —clásico o de MEMO/STHN— podría
  recuperar la señal, porque no está codificada en las etiquetas de origen.
  Confirmarlo requeriría acceso al pipeline del simulador (fuera del alcance
  de este repo) o, más barato, inspeccionar la distribución de profundidad
  de la clase `ArteriesOrVeins` dentro del volumen OCT — no se hizo aquí por
  no ser necesario para responder la pregunta del mandato (la refutación ya
  es suficientemente sólida sin necesidad de explicar el mecanismo causal).
- El test de T2-83/T2-R4 usa **radio máximo 0.04** (fracción del lado) o
  21px de dilatación simétrica — generoso, pero finito. No se puede excluir
  con esta evidencia una correspondencia sistemáticamente desplazada por
  **más** que ese margen en una dirección constante (p.ej. un offset de
  medio milímetro no modelado en la matriz GT). Esto es especulativo y no se
  investigó — no cambia el veredicto porque el control positivo de
  instrumento usa el mismo rango de radios y sí detecta señal con margen.

---

## Resumen — verificado vs. inferido

**Verificado leyendo código**: qué compara cada oráculo (keypoint-vs-máscara
en `verify_task2_enface_convention.py:105-117`; máscara-vs-máscara en
`verify_vessel_signal_candidates.py:96-152`); que T2-82/T2-R2 excluyen
explícitamente el canal de vaso (`train_task2_common.py:411`,
`train_task2_baseline.py:156-162`); los ratios exactos de T2-83
(`experiments/83-t2-enface-convention/oracle_train.md`); los resultados de
`verify_vessel_signal_candidates.py` citados en `ATTACK_LADDER.md:371-401`;
el tamaño de fundus (1024×1024, `THE_MAP.md:147`, `ATTACK_LADDER.md:840-843`).

**Verificado leyendo los papers**: arquitectura de VDD-Reg (§4, p.7-13),
definición de "size ratio" en STHN como ratio de área (§IV-A, p.4), y que
UASTHN trata la autosimilitud como categoría de fallo no resuelta (Fig.1,
p.1-2).

**Inferido, no medido**: la causa mecánica exacta (simulador) de por qué el
volumen OCT y el fundus no corresponden geométricamente en vasos; que la
comparación de tamaños STHN-vs-FIDO usando 15.6%/2.4% para FIDO es correcta
(tomada del contexto de la tarea, no rederivada aquí más allá de la
verificación aritmética cruzada con el tamaño de fundus 1024px y la escala
~160-220px ya reportada en `ATTACK_LADDER.md`).
