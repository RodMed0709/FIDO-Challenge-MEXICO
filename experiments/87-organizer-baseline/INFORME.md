# Informe — paper base de los organizadores (Rohrmoser et al., arXiv 2603.25555)

**Fuente**: `papers/2026_rohrmoser_multimodal-fusion-ophthalmic-surgery-baseline.pdf`
("Towards Comprehensive Real-Time Scene Understanding in Ophthalmic Surgery
through Multimodal Image Fusion", Rohrmoser, Ghazaei, Sommersperger, Navab,
arXiv 2603.25555v1, 26 mar 2026, formato IJCARS, 14 páginas). Leído completo
página por página (texto extraído con `pypdf`, no hay apéndice/material
suplementario incluido en este PDF pese a que el texto lo referencia — ver
§2.3 más abajo).

**Convención de este informe**: todo lo marcado `[PAPER]` es cita textual o
paráfrasis fiel con número de página. Todo lo marcado `[INFERENCIA]` es
razonamiento nuestro, no está en el paper. Todo lo marcado `[REPO]` es un
hecho verificado en nuestro propio código/resultados, para dar contexto de
comparación. Donde el paper no da un dato, se dice explícitamente **"NO
ESPECIFICADO EN EL PAPER"** — no se rellena por analogía.

Este documento ya existía parcialmente cubierto por `literature/02_task1_keypoints.md`
(revisión de literatura previa, 2026-08-15). Este informe relee el PDF desde
cero, corrige un error de esa revisión anterior (confusión µm↔px, ver §5), y
lo actualiza con el estado real del repo a 2026-08-19 (bug de escala de
distancia ya corregido, ver §7B).

---

## 1. Arquitectura exacta

**[PAPER, p.4, Fig.2, §2]** Dos ramas paralelas que confluyen en un módulo de
fusión y, opcionalmente, un módulo recurrente, antes de las cabezas de
predicción:

- **Rama OPMI (primaria)**: backbone **YOLO-NAS-m** (implementación de la
  librería SuperGradients [ref. 21], pesos preentrenados en **COCO** [ref. 25]).
  Genera "multi-scale feature pyramids" desde el *neck* de YOLO-NAS (múltiples
  resoluciones — el paper no dice cuántas escalas exactas, pero YOLO-NAS
  estándar usa 3 niveles de FPN; **NO ESPECIFICADO EN EL PAPER** el número
  exacto para esta variante). A este modelo, sin la rama de iOCT, lo llaman
  **Single-Modality (SM)**.
- **Rama iOCT (secundaria)**: **ResNet-18 modificado** + capa de *adaptive
  pooling* que preserva la resolución lateral. Extrae **M=16 descriptores de
  columna** por B-scan, cada uno de dimensionalidad **C_OCT=16**, "cada uno
  representando A-scans vecinos" — es decir, cada descriptor resume una franja
  lateral del B-scan, preservando la correspondencia espacial entre el perfil
  de profundidad y su posición dentro de la vista de microscopio. M y C_OCT
  son consecuencia de la resolución de entrada y las propiedades de
  downsampling del ResNet-18, no hiperparámetros libres elegidos a mano
  [PAPER, p.8].
- **Entrada**: ambas modalidades se redimensionan a **512×512 px** [PAPER, p.7,
  §3 "Data"]. La entrada de iOCT son **las dos B-scans ortogonales por frame**
  (no el volumen 3D completo, no una proyección en-face) — una alineada con el
  eje del instrumento, la otra perpendicular [PAPER, p.2, Fig.1; p.7].
- **Fusión**: módulo de **cross-attention** aplicado independientemente a cada
  mapa de features de las distintas resoluciones que produce el *neck* de
  YOLO-NAS [PAPER, p.5, §2.1]. Detalle completo en §2 de este informe.
- **Módulo recurrente** (opcional, produce la variante RMM): después de la
  fusión, ver §2 más abajo.
- **Cabezas de predicción**: operan de forma independiente sobre los mapas de
  features multi-escala ya fusionados (y, si aplica, refinados
  temporalmente). Cada cabeza tiene su propia función de pérdida
  específica de tarea [PAPER, p.6, §2.3]:
  1. Clasificación de objeto — *focal loss*.
  2. Regresión de *bounding box* — *distribution focal loss* + *GIoU loss*.
  3. Visibilidad de keypoint — *binary cross-entropy*.
  4. Regresión de keypoint — ***smooth L1*** (regresión directa de
     coordenadas, no heatmap).
  5. Distancia herramienta-tejido (cabeza nueva, no existe en YOLO-NAS
     estándar) — regresión distribucional sobre bins, ver §2 más abajo.

  Todas las pérdidas se combinan en una suma ponderada para entrenamiento
  end-to-end; los pesos se buscaron con Optuna (ver §4). El diseño preserva
  las cabezas originales de detección/keypoint de YOLO-NAS y añade la cabeza
  de distancia como extensión modular [PAPER, p.6].

**[INFERENCIA]** Esta arquitectura resuelve un problema distinto al nuestro en
un aspecto estructural importante: FIDO Task 1 pide **un solo keypoint por
caso** (la punta del instrumento activo) más una distancia, sin necesidad de
detectar/clasificar múltiples instrumentos ni producir cajas delimitadoras.
El paper resuelve **detección multi-instrumento + clasificación + keypoint +
distancia** de forma conjunta sobre vídeo completo. La cabeza de keypoint del
paper vive dentro de un framework de detección densa (predicciones en cada
ubicación espacial del mapa de features, decodificadas con NMS estilo
YOLO) — no es directamente portable a nuestro problema de "un keypoint,
regresión directa desde una imagen ya recortada/completa", pero el resto
(fusión, cabeza de distancia) sí lo es.

---

## 2. La cabeza de distancia (DFL / bins) y el módulo de fusión — lo más copiable

### 2.1 Cross-attention fusion [PAPER, p.5, §2.1, Fig.3]

Problema que resuelve, en palabras del paper: OPMI es una vista en-face de
superficie; iOCT es un corte transversal ortogonal. La alineación
píxel-a-píxel directa es un problema mal planteado (*ill-posed*). Además, la
información de profundidad es espacialmente dispersa y solo relevante en la
posición del instrumento — por eso descartan mezcla global de features y usan
atención selectiva.

- **Queries** = features de píxel de OPMI (el propio mapa de features
  multi-escala del *neck* de YOLO-NAS).
- **Keys/Values** = los M=16 descriptores de columna de iOCT (por B-scan).
- Bloque de atención = **2 bloques apilados** de multi-head attention +
  MLP feed-forward, "similar a la definición de un bloque de encoder
  transformer" [ref. 22, Vaswani et al.] — pero **cross**-attention, no
  self-attention.
- **Positional encoding**: sinusoidal para las queries de OPMI. Cada
  descriptor de columna de iOCT se concatena con un código posicional
  derivado de **su ubicación proyectada en la grilla de OPMI** — es decir, la
  posición del B-scan dentro de la vista de microscopio (dado que un B-scan
  está alineado con el eje del instrumento, cada columna del B-scan
  corresponde a una línea concreta en el espacio de OPMI).
- La salida conserva la resolución espacial y la dimensionalidad de canal que
  requieren las cabezas de predicción downstream — es decir, se suma/inyecta
  de vuelta al mapa de features de OPMI, no lo reemplaza.
- Se aplica **por separado a cada resolución** del *neck* (multi-escala).

**NO ESPECIFICADO EN EL PAPER**: número de cabezas de atención (heads),
dimensión del embedding interno del transformer, dropout, ni si hay
normalización de capa (LayerNorm) explícita — se asume estándar de bloque
transformer pero no se cita.

### 2.2 Cabeza de distancia — regresión distribucional (DFL) [PAPER, p.6-7, §2.3]

Es una adaptación directa del formalismo de regresión de *bounding boxes* de
YOLO-NAS (Distribution Focal Loss, GFLv2 [ref. 23]) aplicado a un escalar
(la distancia), no a 4 coordenadas de caja:

- **Arquitectura de la cabeza**: dos bloques convolucionales (kernel 1×1 y
  3×3) con BatchNorm + ReLU, seguidos de una convolución 1×1 final que
  produce **`reg_max + 1` logits**.
- **`reg_max`**: **NO ESPECIFICADO NUMÉRICAMENTE EN EL PAPER** (el texto
  principal no da el valor; el paper remite a "material suplementario" para
  más detalle de las funciones de pérdida, material que **no está incluido
  en este PDF de 14 páginas** — no se pudo verificar). En YOLO/GFL estándar
  para bbox regression el valor típico es `reg_max=16`, pero **eso es una
  convención de otro dominio (offsets de caja en píxeles de grid), no un
  dato de este paper** — no asumir que aplica igual a un rango de distancia
  en mm.
- **Decodificación**: los `reg_max+1` logits pasan por softmax → distribución
  de probabilidad `p = (p_0, ..., p_regmax)`. El valor esperado es:

  ```
  ŷ = Σ_{i=0}^{regmax} i · p_i
  ```

  Este `ŷ` (en unidades de índice de bin) se **reescala linealmente** al rango
  métrico real `[d_min, d_max]` — el paper no da la fórmula de reescalado
  explícita, pero por construcción es una interpolación lineal
  `distancia_mm = d_min + ŷ/reg_max · (d_max - d_min)`.
- **Rango de distancia usado en el paper**: **`d_min = -1 mm`, `d_max = 6 mm`**
  [PAPER, p.8, §"Experimentation Setup"]. El rango negativo es intencional:
  admite que la punta esté *dentro* del tejido (inyección subretiniana es uno
  de los dos tipos de cirugía simulados).
- **Pérdida**: *distribution focal loss* [ref. 23, Li et al. 2020, GFLv2],
  que empuja al modelo a aprender una distribución de probabilidad
  **afilada (sharp)** centrada en la distancia verdadera — no simplemente a
  minimizar el error del valor esperado.
- **Certeza (bonus, no cuesta nada extra)**: para el bin más probable
  `i* = argmax_i p_i`, la certeza se calcula como:

  ```
  certeza = p_{i*} + max(p_{i*-1}, p_{i*+1})
  ```

  Es decir, la masa de probabilidad concentrada en el bin ganador y su vecino
  más fuerte. El paper reporta que esta certeza está **calibrada**: mayor
  certeza predicha correlaciona con menor error real (ver §5, Fig.4 y
  "certain dMAE").

**[INFERENCIA]** Esta cabeza no depende de que exista un framework de
detección tipo YOLO — es un módulo autocontenido (conv → softmax → esperanza
ponderada → reescalado lineal) que se puede injertar sobre **cualquier**
extractor de features de B-scan (p. ej. nuestro encoder de U-Net de
segmentación, o un CNN dedicado sobre los 2 B-scans), sin necesidad de
adoptar YOLO-NAS ni la cabeza de keypoint del paper. Es la pieza más
directamente portable de todo el paper a nuestro código.

### 2.3 Módulo recurrente [PAPER, p.6, §2.2] — para contexto, no aplica a Competition

- Tras la fusión: *average pooling* del mapa fusionado → grilla compacta de
  descriptores regionales → una **unidad recurrente (GRU o LSTM)** procesa la
  secuencia temporal de esos descriptores a través de frames consecutivos →
  se reinterpola a la resolución original → se combina con la representación
  fusionada (el paper no especifica si es suma, concatenación o gating —
  **NO ESPECIFICADO**).
- Entrenado con secuencias de **16 frames consecutivos**.
- **Pérdida auxiliar**: *cosine contrastive loss* entre features derivadas de
  iOCT corrupto (`f_c`) e iOCT intacto (`f_n`): `L_cos = 1 - cos(f_c, f_n)` —
  fuerza a que la representación temporal sea robusta/invariante a la
  corrupción del iOCT.
- Durante entrenamiento: **4 frames consecutivos + 4 frames aleatorios** de
  cada secuencia de 16 tienen su embedding de iOCT reemplazado por vectores
  aleatorizados, simulando datos poco confiables.
- Arquitectura del GRU/LSTM: **NO ESPECIFICADO** (número de capas — sólo se
  menciona "4-layer GRU/LSTM" en el pie de Tabla 2 —, tamaño oculto, etc. Sí
  se aclara: **4 capas**, `p.11 Tabla 2`).

**[INFERENCIA]** Este módulo no es relevante para nosotros ahora mismo: FIDO
Task 1 se evalúa caso por caso, no en secuencias de vídeo, así que la
recurrencia temporal no tiene dónde aplicarse en la interfaz de submission
actual (`inference(task_id, oct_volume, opmi_image, model)` recibe un caso a
la vez). Además, el propio paper muestra que este módulo **empeora** las
métricas frame-a-frame respecto al modelo estático (ver §6) — doble motivo
para no perseguirlo.

---

## 3. Preprocesamiento de iOCT [PAPER, p.7, §3 "Data"; p.5, §2]

- **Entrada**: **las dos B-scans ortogonales** centradas en la punta del
  instrumento, una alineada con el eje del instrumento y la otra
  perpendicular (ver Fig.1, p.2). **No usan el volumen 3D completo, no usan
  proyección en-face.** Esto es una diferencia frente al enfoque de en-face
  que exploramos para Task 2 — aquí el equivalente al "en-face" sería la
  imagen OPMI misma, que ya viene aparte.
- **Resolución**: 512×512 px (mismo tamaño que OPMI tras el resize).
- **Normalización**: "ambas modalidades pasan por normalización y
  aumentación con variaciones moderadas de color, ruido y niebla (*fog*) para
  simular condiciones de imagen realistas" [PAPER, p.7]. **No se especifica
  la fórmula de normalización** (¿por canal, min-max, z-score con qué
  media/std?) — **NO ESPECIFICADO EN EL PAPER**.
- **Artefactos sintéticos modelados explícitamente**: sombreado del
  instrumento (*instrument shadowing*, oculta la anatomía retiniana bajo el
  instrumento) y **espejado a lo largo del cero de retardo del OCT (mirroring
  along the OCT zero-delay)**, que surge del conjugado complejo extraído
  durante la reconstrucción de señal en sistemas OCT reales de dominio
  espectral/swept-source [PAPER, p.7]. Esto es una pista de qué artefactos
  espera el simulador que reproduzcamos/toleremos si el dataset de FIDO viene
  del mismo simulador (SynthesEyes GmbH — ver Declaraciones, p.11: **M.S. y
  N.N. son accionistas de SynthesEyes GmbH, que proveyó los datasets
  sintéticos** — mismo proveedor de datos que probablemente FIDO).
- **Embedding de iOCT**: ver §1/§2.1 — ResNet-18 modificado, M=16
  descriptores × C_OCT=16 dims, por B-scan.

**[INFERENCIA]** El dato de "mirroring along OCT zero-delay" es
potencialmente accionable: si el dataset de FIDO reproduce el mismo
artefacto, un B-scan podría tener una réplica especular parcial de la
anatomía cerca del borde superior/inferior. Vale la pena una inspección
visual rápida de B-scans reales de FIDO para confirmar si aparece, antes de
asumir que el B-scan es "limpio".

---

## 4. Entrenamiento [PAPER, p.8, §3 "Experimentation Setup"]

| Hiperparámetro | Valor | Cita |
|---|---|---|
| Backbone base | YOLO-NAS-m (SuperGradients), pesos COCO-preentrenados | p.8 |
| Épocas (estático) | **50** | p.8 |
| Épocas (temporal/RMM) | **120** | p.8 |
| Optimizador | **AdamW** | p.8 |
| Schedule de LR | **coseno** | p.8 |
| Learning rate | **1.2726e-4** | p.8 (elegido por Optuna) |
| Búsqueda de hiperparámetros | **Optuna**, optimizando conjuntamente mAP50 y dMAE; incluye pesos de las pérdidas y el LR | p.8 |
| Hardware | **1× NVIDIA RTX 3090** | p.8 |
| Secuencia temporal (RMM) | 16 frames consecutivos | p.8 |
| Corrupción de entrenamiento (RMM) | 4 frames consecutivos + 4 aleatorios con iOCT reemplazado por vectores aleatorizados, por secuencia de 16 | p.8 |
| Batch size | **NO ESPECIFICADO EN EL PAPER** | — |
| Augmentación exacta (tipos, magnitudes) | Solo cualitativo: "color, noise, fog variations, moderate" | p.7 |
| Tamaño de dataset | 20 vídeos, **69,134 frames**, 45-60 FPS | p.7 |
| Tipos de cirugía | Membrane peeling + subretinal injection | p.7 |
| Clases de instrumento | 4 (2 tipos de fórceps, cánula quirúrgica, endoiluminador) | p.7 |
| Split | **17:3 vídeos** train:val (a nivel de vídeo, no de frame) | p.7 |
| Resolución de entrada | 512×512 px (ambas modalidades) | p.7 |
| Subsampling (frame-wise) | cada **30avo** frame | p.7 |
| Subsampling (temporal) | cada **10avo** frame | p.7 |
| Rango de distancia (cabeza DFL) | d_min=-1mm, d_max=6mm | p.8 |

**[INFERENCIA]** El split **a nivel de vídeo** (17:3, no a nivel de frame) es
exactamente la misma disciplina que ya aplicamos con GroupKFold por escenario
(`THE_MAP.md`, regla transversal 1) — el paper confirma que es la práctica
correcta en este dominio exacto, no una precaución nuestra excesiva.

---

## 5. Métricas reportadas — definición exacta [PAPER, p.8, §"Metrics"; Tabla 1, p.9]

| Métrica | Definición textual | Unidad |
|---|---|---|
| `mAP50` | mean Average Precision @ IoU=0.50 (detección de instrumento) | % |
| `kp_dist` | "average pixel distance between predicted and ground-truth keypoints" | **px**, a resolución de entrada 512×512 (inferido del contexto — el paper no repite "a 512px" junto a la definición de la métrica, pero es la única resolución mencionada en todo el pipeline) |
| `dMAE` | mean absolute error de la distancia herramienta-tejido, sobre **todas** las distancias | **µm** |
| `dMAE0-1` | igual que dMAE, pero restringido a distancias reales **< 1 mm** | µm |
| `certain dMAE` | dMAE restringido a predicciones con certeza (§2.2) **> 90%** | µm |

**Tabla 1 — comparación SM vs MM vs RMM, sin recurrencia salvo donde se indica** [PAPER, p.9]:

| Modelo | mAP50 [%] | kp_dist [px] | dMAE [µm] | dMAE0-1 [µm] | certain dMAE [µm] |
|---|---:|---:|---:|---:|---:|
| SM (solo OPMI) | 94.27 ± 0.41 | 8.35 ± 1.32 | 480.93 ± 54.91 | 284.01 ± 89.44 | 142.99 ± 25.4 |
| MM (OPMI+iOCT) | 95.79 ± 0.58 | 7.93 ± 0.47 | 128.32 ± 20.03 | 33.05 ± 2.38 | 53.74 ± 13.0 |
| RMM (GRU) | 94.92 ± 0.28 | 9.65 ± 0.42 | 332.91 ± 48.88 | 135.44 ± 37.28 | 61.78 ± 10.73 |
| RMM (LSTM) | 94.60 ± 0.29 | 9.17 ± 0.46 | 351.59 ± 34.66 | 141.85 ± 67.60 | 62.88 ± 14.54 |

Significancia estadística [PAPER, p.9-10]:
- keypoint SM vs MM: **Welch's t-test p=0.525 → NO significativo**. La fusión
  con iOCT no mueve la localización del keypoint.
- mAP50 SM vs MM: p<0.001, Cohen's d=2.99 (significativo, mejora absoluta
  pequeña: +1.52 pp).
- dMAE global SM vs MM: p<0.001, Cohen's d=8.92, **reducción del 73%**
  (480.93→128.32 µm).
- dMAE0-1 (rango <1mm) SM vs MM: p=0.002, **reducción del 88%**
  (284.01→33.05 µm).
- certain dMAE SM vs MM: p<0.0001, Cohen's d=4.57 (143→54 µm).

**Tabla 2 — robustez del RMM (GRU/LSTM) a corrupción de iOCT durante
inferencia, sobre una secuencia de peeling, media±std sobre 5 semillas**
[PAPER, p.10]:

| #frames corruptos | dMAE [µm] | masked dMAE [µm] | dMAE0-1 [µm] | masked dMAE0-1 [µm] |
|---|---:|---:|---:|---:|
| GRU, 0 | 191.80 ± 11.4 | — | 62.31 ± 17.1 | — |
| GRU, 4 | 232.19 ± 14.4 | 458.82 ± 59.3 | 89.59 ± 25.0 | 262.40 ± 69.3 |
| GRU, 8 | 314.01 ± 13.4 | 585.18 ± 30.0 | 177.71 ± 23.3 | 532.43 ± 32.3 |
| LSTM, 0 | 222.89 ± 24.3 | — | 65.50 ± 9.4 | — |
| LSTM, 4 | 264.00 ± 24.8 | 471.79 ± 59.8 | 99.04 ± 15.0 | 305.29 ± 71.5 |
| LSTM, 8 | 348.13 ± 27.7 | 604.74 ± 96.4 | 193.48 ± 35.5 | 574.80 ± 129.6 |

"masked" = calculado solo sobre los frames con iOCT corrupto en esa corrida.

Runtime: **22.5 ms/frame (MM)** vs **18.0 ms/frame (SM)** — real-time (~44 Hz
vs ~55 Hz) [PAPER, p.9-10]. **NO ESPECIFICADO** el runtime de RMM.

### Corrección a la revisión de literatura previa (µm vs px)

`literature/02_task1_keypoints.md` (2026-08-15) razonaba que "ni siquiera el
SOTA publicado entra cómodamente bajo el umbral de 10 (µm)" comparando
directamente los µm del paper contra los umbrales enteros 0..10 de la métrica
de FIDO — **eso mezclaba unidades sin convertir** (el propio archivo lo
señala como riesgo sin resolver en la línea 316). **`RULES_OF_ENGAGEMENT.md`
§8 confirma que el ground truth de distancia de FIDO está en píxeles reales
de B-scan**, y `src/fido/data/task1.py` (corregido 2026-08-19, `[REPO]`) fija
el factor de conversión verificado contra el scoring oficial:
**`GROUND_TRUTH_DISTANCE_SCALE = 4000 µm / 512 px = 7.8125 µm/px`** (eje
axial). Aplicando ese factor a los números del paper:

| Métrica del paper | En µm | Convertido a px de B-scan (÷7.8125) |
|---|---:|---:|
| dMAE global (MM) | 128.32 µm | **≈16.4 px** |
| dMAE0-1 (MM, régimen <1mm) | 33.05 µm | **≈4.2 px** |
| dMAE global (SM) | 480.93 µm | **≈61.6 px** |

**[INFERENCIA]** Con los umbrales de AUC de FIDO siendo enteros 0..10 px, el
dMAE global del SOTA publicado (≈16.4 px de error medio) **sí** quedaría en
gran parte fuera de la ventana de 10 px si la distribución de error fuera
uniforme — pero su error en el régimen cercano (≈4.2 px) cae cómodamente
**dentro** de la ventana. Esto es una corrección real de expectativa (no del
paper, es aritmética nuestra): el SOTA publicado sí es alcanzable/competitivo
en umbrales bajos **específicamente para el régimen de distancias cortas**,
que es exactamente donde la literatura de nuestro propio proyecto
(`literature/02_*.md`, distribución de distancias GT con mediana ≈169 µm ≈
21.6 px) dice que vive la mayoría de la masa de datos. No es una conclusión
tan pesimista como la que dejó la revisión anterior.

---

## 6. Ablations — qué aporta cada componente [PAPER, p.9-11, §4]

| Componente añadido | Efecto en mAP50 | Efecto en kp_dist | Efecto en dMAE |
|---|---|---|---|
| iOCT + cross-attention (SM→MM) | +1.52pp, significativo (d=2.99) | **sin cambio significativo** (p=0.525) | **-73%** global, **-88%** en <1mm (ambos muy significativos) |
| Módulo recurrente (MM→RMM, bajo condición de corrupción de evaluación) | -0.87pp (GRU) | **empeora**: 7.93→9.65px (GRU), 9.17px (LSTM) | **empeora**: 128.32→332.91µm (GRU), 351.59µm (LSTM) |
| GRU vs LSTM (dentro de RMM) | GRU levemente mejor | GRU mejor (9.65 vs 9.17 — en realidad LSTM da kp_dist ligeramente MEJOR, pero peor en todo lo demás) | GRU consistentemente mejor en distancia |
| Certeza >90% (filtro de calidad, no un componente arquitectónico) | — | — | **certain dMAE** 143→54µm (SM→MM), confirma que la certeza predicha correlaciona con precisión real |

Conclusión explícita del paper sobre la fusión multimodal [PAPER, p.9-10]: la
ganancia de iOCT es asimétrica — **prácticamente nula en detección/keypoint,
enorme en profundidad**. Los autores lo atribuyen a que "OPMI captura el
contexto quirúrgico global mientras que iOCT contribuye detalle geométrico
localizado esencial para razonamiento de profundidad preciso" [PAPER, p.10].

Sobre el módulo recurrente, el paper es explícito en que **no vale la pena
para el caso limpio**: incluso con 0 frames corruptos, el RMM (GRU) da
dMAE=191.80µm — peor que el MM estático (128.32µm) [Tabla 2 vs Tabla 1]. El
recurrente solo compra **resiliencia moderada** ante corrupción de corto
plazo, a costa de rendimiento en el caso limpio, y **colapsa** (no recupera
el nivel del SM) cuando el iOCT está persistentemente corrupto — los autores
lo nombran explícitamente como una forma de **"modality collapse"**
[ref. 26, Chaudhuri et al. 2025], sobre-dependencia del modelo en el stream
de iOCT.

---

## 7. Limitaciones declaradas por los propios autores [PAPER, p.10-11, discusión y conclusión]

1. **Dataset sintético, no clínico.** Explícitamente: "Rather than solving
   the Sim2Real gap, this work focuses on exploring and evaluating
   methodologies for multimodal fusion" [p.10]. No reclaman robustez bajo
   todas las condiciones intraoperatorias reales.
2. **No cubre todos los casos quirúrgicos con patologías complejas** [p.10].
3. **Datos reales serían más desafiantes** por mayor ruido y artefactos más
   severos [p.10].
4. **Conflicto de interés declarado**: M. Sommersperger y N. Navab son
   accionistas de **SynthesEyes GmbH**, la empresa que proveyó el dataset
   sintético [p.11, Declarations]. Relevante para calibrar cuánto pesar sus
   cifras como "techo alcanzable" vs. cifras optimizadas sobre datos propios
   de la empresa de los propios autores.
5. **Modality collapse del módulo recurrente** bajo corrupción sostenida de
   iOCT — el modelo se vuelve excesivamente dependiente del iOCT y no
   recupera el nivel de SM cuando el iOCT falla de forma persistente [p.10].
6. **No hay ground truth de datasets clínicos multimodal anotados
   disponibles** actualmente — motivo explícito para usar datos sintéticos
   [p.7, §3 "Data"]: registración precisa iOCT↔microscopio y tracking de
   instrumento para posicionar los B-scans no están aún integrados en
   dispositivos comerciales, aunque son técnicamente viables por separado.
7. Trabajo futuro que ellos mismos proponen: mecanismos de fusión adaptativos
   que evalúen y balanceen explícitamente la contribución de cada modalidad,
   para lograr comportamiento multimodal más robusto y confiable [p.10-11].

---

## 8. Respuestas a las dos preguntas que importan

### A) Nuestro techo de keypoint ~0.853 con tres backbones distintos — ¿qué hacen ellos que nosotros no, y que podría explicarlo?

**Primero, lo que el propio paper descarta como explicación**: su propia
ablation muestra que añadir iOCT vía cross-attention **no mueve el keypoint
de forma significativa** (p=0.525, kp_dist 8.35→7.93px, un 5% de mejora
dentro del ruido). Y sus tres variantes — SM (solo CNN detector), MM
(+ cross-attention transformer con iOCT), RMM (+ recurrencia GRU/LSTM) — dan
kp_dist en la misma banda estrecha, **7.93 a 9.65 px**, pese a ser
arquitecturas radicalmente distintas (detector denso vs fusión
transformer vs temporal). Esto **replica exactamente el patrón que ya
observamos nosotros**: tres backbones (CNN, ResNet-18+FPN, DINOv2) plateau en
~0.853 de keypoint_auc. La evidencia externa, con un equipo con más recursos,
mejor pretraining (COCO detección) y arquitecturas mucho más sofisticadas,
apunta en la misma dirección que nuestros tres experimentos: **el cuello de
botella del keypoint no es la capacidad del extractor de features, ni el uso
(o no) de fusión multimodal.**

Con esa base, priorizado por retorno esperado / esfuerzo:

1. **[Retorno incierto, esfuerzo bajo] Investigar la brecha local→Codabench
   (0.8539 → ~0.774) antes que seguir tocando arquitectura.** Esta brecha
   (0.08 de AUC) es **16x más grande** que la diferencia entre nuestros tres
   backbones (0.0005-0.05). El paper no explica esta brecha específica (usa
   su propio split interno, no compara contra un leaderboard externo), pero
   sí confirma con fuerza que el eje "arquitectura del extractor" está
   agotado — así que el margen que queda casi seguro vive en
   distribución de datos (overfit al split local, augmentación insuficiente,
   normalización distinta en el entorno de Codabench) y no en la elección de
   backbone. **No hay nada más que exprimir del paper aquí** — esto es
   trabajo de diagnóstico propio, no una receta que copiar.
2. **[Retorno bajo, esfuerzo medio] NO perseguir cross-attention OPMI↔iOCT
   para el keypoint.** El propio SOTA lo probó con recursos serios y no le
   dio significancia estadística. Gastar presupuesto de Competition (quedan
   ~2 días) en esto tiene expectativa negativa según la evidencia del propio
   paper base.
3. **[Retorno bajo, esfuerzo alto] NO perseguir un módulo recurrente/temporal
   para el keypoint.** El paper muestra que **empeora** kp_dist (7.93→9.17-9.65
   px) incluso sin corrupción. Coincide con que FIDO Task 1 se evalúa caso a
   caso (`inference()` recibe un caso, no una secuencia), así que ni siquiera
   hay una interfaz natural para explotarlo en Competition.
4. **[Posible pero no verificado, esfuerzo bajo-medio] Pretraining orientado a
   localización/detección en vez de clasificación.** Diferencia real entre su
   setup y el nuestro: ellos parten de pesos **COCO-preentrenados** (tarea de
   detección/localización), nosotros probamos ImageNet-clasificación
   (ResNet-18) y self-supervised genérico (DINOv2) — ninguno de los dos es
   pretraining específicamente de localización densa. **Esto no está
   validado por una ablation en el paper** (no comparan COCO-pretrained vs
   scratch para el keypoint específicamente) — es una diferencia observada,
   no una receta confirmada. Dado que ya tenemos el scaffolding de
   ResNet-18+FPN, probar pesos de un backbone preentrenado en una tarea de
   detección (p. ej. weights de un modelo de object detection COCO en vez de
   clasificación ImageNet) es barato de intentar, pero la expectativa debe
   calibrarse baja: dado que YA demostramos que el backbone no es el cuello
   (tres arquitecturas, mismo techo), es poco probable que cambiar solo el
   *origen* del pretraining rompa ese techo.

**Conclusión para A**: el paper no ofrece una palanca de arquitectura que
explique nuestro techo — de hecho, es evidencia adicional (la cuarta,
contando nuestras tres) de que **el techo de ~0.85 en keypoint_auc es del
planteamiento del problema (definición del target, ruido de anotación,
resolución de la métrica de umbrales 0..10, o límite de precisión alcanzable
en este dominio), no de la arquitectura**. La recomendación de mayor
retorno esperado es dejar de invertir en backbones/fusión para keypoint y
mover el esfuerzo a diagnosticar la brecha local↔Codabench y a la
componente de distancia (ver B).

### B) Nuestra distancia da 0.0900 con segmentación+fórmula lineal — ¿vale la pena migrar a bins/DFL?

**Contexto actualizado que cambia la pregunta** [REPO, `RESULTS.md` línea 241,
2026-08-19]: el `distance_auc=0.0900` de la submission `r01` en Codabench ya
se diagnosticó — **no era un problema del modelo ni del enfoque geométrico**,
sino un **sesgo sistemático de escala del 28%**: el ajuste lineal se calibró
contra `GT/10` cuando el valor real es `GT/7.8125`. Corregido a
`dist = 0.9370·pixel_gap + 3.4277`. Un control aislado en 5 casos del Mock
Test (línea 245) mostró que el reescalado por sí solo sube `distance_auc` de
`0.0000` (bajo el scorer corregido) a `0.1333` — dirección confirmada,
magnitud no (n=5, y el Mock Test no se usa para seleccionar hiperparámetros
por `CONSTITUTION.md` §3). Además, un entrenamiento local más reciente
(T1-R5 v2, `NOW.md` línea 140) da `val_distance_auc=0.6113` en época 10 sobre
12,127 casos — aunque **inflado**, porque `evaluate()` descarta casos sin gap
de segmentación medible, así que no es directamente comparable al número de
Codabench.

Dado esto, la pregunta real no es "¿el enfoque geométrico está agotado?" —
locally parece tener mucho recorrido sin tocar la arquitectura, solo
arreglando el bug de escala. La pregunta es: **¿qué aporta específicamente el
enfoque DFL del paper que el enfoque geométrico actual no puede dar, y
justifica el esfuerzo de construirlo ahora?**

**Lo que el DFL soluciona que la segmentación geométrica no puede, por
diseño**:
- **Robustez estructural a fallos de segmentación.** El pipeline geométrico
  actual necesita segmentar correctamente **tanto** la cánula **como** el ILM
  en el mismo B-scan para poder medir un "gap" — si cualquiera de las dos
  falla (oclusión, sombreado del instrumento, el propio artefacto de
  *shadowing* que el paper describe en §3), no hay predicción geométrica
  posible. `NOW.md` línea 140 confirma que esto ya es un problema real hoy:
  `evaluate()` **descarta** los casos sin gap medible en vez de dar una
  predicción para ellos. Cualquier caso descartado en evaluación local pero
  no en Codabench (donde SIEMPRE hay que devolver un número) probablemente
  cae en el fallback/sentinel actual y arrastra el AUC hacia abajo con fuerza
  — exactamente el tipo de cola que el AUC (promedio de aciertos a
  umbral 0..10) castiga más. La cabeza DFL, al ser una regresión directa
  desde features de imagen (no una medición geométrica en dos pasos),
  **siempre produce una distribución**, sea cual sea la calidad de la
  segmentación subyacente.
- **Certeza calibrada "gratis"** — útil sobre todo para decisiones internas
  (p. ej., cuándo confiar más en la rama geométrica vs una rama aprendida, o
  como señal para debug), aunque el formato de submission de FIDO
  (`RULES_OF_ENGAGEMENT.md` §8: se devuelve un único float) no tiene un
  mecanismo para explotar la certeza en el score directamente.
- **Maneja de forma natural el régimen de distancias negativas** (herramienta
  dentro del tejido) si FIDO tiene casos así — a verificar contra la
  distribución real del GT de FIDO (no confirmado en este informe).

**Estimación de impacto** (distancia pesa 0.30 del score total, keypoint
0.70): con `distance_auc` moviéndose desde el ~0.09 original hacia algo en el
rango que ya sugiere nuestro propio T1-R5 v2 local (orden de magnitud 0.3-0.6,
con la salvedad de que ese número está inflado), la ganancia en el score
final es de **+0.3×(0.6-0.09) ≈ +0.15 puntos** en el caso optimista, o
**+0.3×(0.25-0.09) ≈ +0.05** en un caso conservador — comparable o mayor que
cualquier ganancia realista de keypoint_auc dado el techo ya demostrado
(mover keypoint_auc medio punto porcentual pesa 0.7×0.005≈0.0035, dos órdenes
de magnitud menos). **La distancia sigue siendo, con mucha diferencia, el
lugar de mayor retorno por hora invertida en Task 1** — esto ya estaba
identificado en `RESULTS.md` (línea 243: "el margen del proyecto está aquí,
no en Task 2") y el paper lo confirma desde otro ángulo: es donde la fusión
multimodal SÍ tiene un efecto brutal y estadísticamente inequívoco (73-88% de
reducción de error), a diferencia del keypoint.

**Recomendación concreta, con el reloj en la cabeza** (quedan ~2 días de fase
Competition según `THE_MAP.md`):

- **Ahora (Competition, próximas horas)**: **NO** reescribir la cabeza de
  distancia a DFL. Priorizar cerrar y **subir** la corrección de escala ya
  hecha (`0.9370·pixel_gap + 3.4277`) y confirmar el número real en
  Codabench — es el cambio de mayor ROI inmediato y ya está implementado, sin
  riesgo de arquitectura nueva sin probar contra el scorer real.
- **Final Round (20 días, si el enfoque geométrico se queda corto o muestra
  la cola de casos sin gap medible como problema dominante)**: implementar la
  cabeza DFL tal como la especifica el paper (§2.2 de este informe) como
  **head adicional sobre el encoder de B-scan que ya existe** (el U-Net de
  T1-R5), no como reemplazo del pipeline geométrico sino como
  alternativa/ensemble — esto es exactamente lo que ya estaba planeado en
  `ATTACK_LADDER.md` T1-R5 ("si no [R²>0.95]: regresión distribucional (DFL
  sobre bins)"), y este informe lo confirma como la ruta correcta con
  detalle de implementación citado del paper. Elegir `reg_max` y el rango
  `[d_min, d_max]` a partir de la distribución real del GT de distancia de
  FIDO (no copiar `-1..6mm` del paper sin verificar que aplica a nuestra
  escala en píxeles de B-scan).

---

## 9. Resumen de brechas explícitas (todo lo que el paper NO dice)

Para que quede como checklist, todo lo marcado **NO ESPECIFICADO** arriba,
junto:

- Número exacto de escalas del *neck* de YOLO-NAS-m usadas en la fusión.
- Hiperparámetros internos del bloque de cross-attention (heads, dim,
  dropout, normalización).
- Valor numérico de `reg_max` en la cabeza de distancia.
- Fórmula exacta de reescalado bin→métrico (se infiere lineal, no se cita
  la fórmula).
- Batch size de entrenamiento.
- Composición exacta de la augmentación (tipos, magnitudes, probabilidades).
- Arquitectura interna del GRU/LSTM (dimensión oculta; solo se sabe que son
  4 capas).
- Mecanismo exacto de combinación entre features fusionadas y
  temporalmente-refinadas (suma, concat, gating).
- Runtime de la variante RMM (solo se da el de SM y MM).
- Contenido del "material suplementario" referenciado para las funciones de
  pérdida — no está incluido en este PDF de 14 páginas.

---

## 10. Archivos de referencia usados para contexto del repo

- `RULES_OF_ENGAGEMENT.md` §8-9 — formato de retorno y fórmula de score de
  Task 1.
- `src/fido/data/task1.py` — `GROUND_TRUTH_DISTANCE_SCALE = 7.8125 µm/px`
  (corregido 2026-08-19).
- `RESULTS.md` líneas 54-56, 73, 112-117, 241-245 — historial de submissions
  y diagnóstico del bug de escala de distancia.
- `NOW.md` líneas 86-140 — estado más reciente de T1-R5 v2.
- `ATTACK_LADDER.md` (sección T1-R5) — plan ya existente para DFL como
  fallback.
- `literature/02_task1_keypoints.md` — revisión de literatura previa sobre
  este mismo paper (2026-08-15); este informe corrige su confusión µm/px
  (§5 arriba) y actualiza con el estado del repo a 2026-08-19.
