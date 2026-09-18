# Ronda 2 de literatura — Task 2, registración sin correspondencia de apariencia

Fecha: 2026-08-19. Objetivo: cuatro frentes no explorados en la síntesis previa
(`papers/INDEX.md`, ronda 1), todos evaluados contra el hecho ya medido de que **la
vasculatura no corresponde entre OCT y fundus en la posición GT** (ratio
observado/null 0.78–1.17 sobre 1214 casos, memoria `fido-task2-appearance-oracle-negative`)
y de que **el instrumento sí corresponde** (enriquecimiento 3.0–3.6x sobre 312 puntos).
6 papers nuevos descargados, indexados en `papers/INDEX.md` § Ronda 2. Ningún subagente
lanzado; búsqueda y lectura hechas directamente (WebSearch/WebFetch/pdftotext).

Veredicto ejecutivo: **de los cuatro frentes, ninguno entrega una receta lista para
implementar sin trabajo original.** A sobrevive solo como principio (ya validado por
nosotros), no como método publicado aplicable a un par de imágenes estático. B tiene una
respuesta técnica precisa y negativa: DPCN/DPCN++ no es keypoint-free en el sentido que
importa aquí. C no encontró ningún método publicado que registre retina cruzando
modalidades sin apoyarse en vasos, bordes o intensidad correlacionada — todos mueren por
la misma razón que ya mató la vía de vasos. D confirma que la aumentación exacta ya
planeada es la apuesta correcta, y aporta una alternativa arquitectónica real si esa
aumentación no basta.

---

## Frente A — Registración guiada por el instrumento quirúrgico

**Pregunta:** ¿existe un método publicado que registre iOCT↔microscopio usando el
instrumento como landmark compartido, aplicable a un único par de imágenes estático (sin
video, sin cinemática del robot)?

**Lo que se encontró.**

- **Zhou et al. 2018** (`2018_zhou_hand-eye-calibration-oct-robotic-eye-surgery.pdf`,
  grupo Nasseri/Navab, TUM) resuelve exactamente el problema conceptual — la punta de
  aguja como diana compartida entre OCT y otro sistema de coordenadas — con error de
  calibración de 9.2–7.0 µm. Pero el "otro sistema de coordenadas" es la **cinemática del
  robot**, no una imagen de microscopio, y la calibración se hace con **desplazamientos
  micrométricos controlados en el tiempo**: el robot mueve la aguja a varias posiciones
  conocidas, cada una se detecta en un volumen OCT 3D, y de esas correspondencias
  múltiples se resuelve la transformación. Es un procedimiento de calibración activo,
  no una inferencia de pose desde un par de imágenes congelado.
- Búsquedas adicionales (integración microscopio-iOCT, registración diagnóstica↔iiOCT
  por curvatura, AR intraoperatoria) confirman el mismo patrón: **todo lo publicado
  sobre "instrumento como ancla" en cirugía ocular usa o bien cinemática del robot, o
  bien continuidad temporal de video** (tracking frame a frame, filtros de Kalman,
  IPCC iterativo sobre 50 frames). Ninguno resuelve "un único frame OCT + un único frame
  de microscopio, sin más contexto" — que es exactamente el régimen de Task 2 (un
  snapshot, sin secuencia).
- El paper de landmarks de herramienta de Probst et al. 2017 (RA-L, no descargado
  — no involucra OCT) calibra el microscopio estéreo contra la cinemática del robot con
  keypoints de herramienta detectados por deep learning; mismo patrón: la segunda
  modalidad es cinemática, no una segunda imagen.
- **Vasconcelos et al. 2016** (`2016_vasconcelos_similarity-registration-ultrasound-calibration.pdf`)
  es el hallazgo más útil de este frente, aunque indirecto: da solución cerrada mínima
  para calibración con aguja trackeada contra ultrasonido 2D/3D — 2 correspondencias
  línea-línea o 4 punto-línea para resolver una similitud (rotación+escala+traslación).
  La geometría es 3D-línea, no transfiere literalmente, pero **confirma formalmente**
  que el problema de "resolver una transformación de similitud desde muy pocas
  correspondencias de una herramienta" tiene solución cerrada estándar — que es
  exactamente Procrustes/Umeyama con 2 puntos, ya adoptado en nuestro diseño de Task 2
  (memoria `fido-task2-geometry`).

**Aplicable a nuestro caso? Con matices.** El *principio* de A (usar el instrumento
como ancla) ya está validado empíricamente por nosotros, mejor que por cualquier paper
encontrado (312 puntos, enriquecimiento 3.0–3.6x, medido con la convención de
orientación correcta). Pero **no existe literatura que resuelva nuestro régimen
exacto** (single-frame, sin cinemática, sin video): todo lo publicado necesita una señal
adicional que no tenemos. Esto no es un método a adoptar — es una vía a construir
desde cero, con una ventaja importante: **Task 1 ya entrena un detector de la punta del
instrumento en la imagen de microscopio** (GT de Task 1 es `[x, y, distancia]` en esa
misma imagen). Lo que falta es el lado OCT: un detector del instrumento en la
**proyección en-face** (no en el B-scan, que es lo que hace Task 1 del lado OCT). Dado
que ya existe segmentación de instrumento por B-scan (`InstrumentInOCT`, usada en
T1-R5), proyectarla a en-face (mismo `mean(axis=1)` que ya se usa para vasos) es
barato de intentar.

**Qué requeriría implementarlo:**
1. Detector de punto(s) del instrumento en la proyección en-face del volumen OCT
   (reusar/adaptar el pipeline de segmentación de Task 1, proyectado).
2. Reusar (o entrenar una cabeza gemela de) el detector de instrumento de Task 1 sobre
   la imagen de microscopio/fundus.
3. Resolver la similitud de 4 DOF con Procrustes/Umeyama desde 1–2 correspondencias del
   instrumento (matemática ya conocida, sin necesidad de ningún paper adicional) —
   probablemente con regularización/prior aprendido para los casos dones el instrumento
   no aparece dentro de la huella del OCT (77% de los casos, según la medición negativa
   de 2026-08-18) o cae fuera del fundus.
4. Validar contra el techo teórico: la medición negativa de 2026-08-18 (con la
   convención de orientación *antes* de corregirla) encontró instrumento dentro de la
   huella del OCT en solo el 23% de los casos; no hay todavía una cifra de cobertura
   recalculada con la convención corregida (la que sí dio 3.0–3.6x de enriquecimiento
   sobre 312 puntos). Sea cual sea el número exacto, es razonable esperar que esta vía
   **no cubra todos los casos** — necesita combinarse con otra señal para el resto, o
   servir de ancla parcial dentro de un modelo más grande (p. ej. como término de loss
   auxiliar, no como único predictor).

**Veredicto A: el principio sobrevive, la literatura no aporta un método listo.** No
hay nada que "adoptar"; hay una arquitectura a diseñar desde el hallazgo propio, con el
riesgo explícito de que cubre una minoría de casos.

---

## Frente B — Correlación de fase log-polar (DPCN/DPCN++): ¿keypoint-free de verdad?

Leídos completos por extracción de texto (`pdftotext`) — no solo el abstract — los dos
papers ya descargados en ronda 1: `2020_chen_deep-phase-correlation-heterogeneous-matching.pdf`
(DPCN, CoRL 2020) y `2022_chen_dpcnpp-differentiable-phase-correlation.pdf` (DPCN++).

**Mecanismo exacto.** DPCN no es un matcher de puntos: entrena dos U-Net (uno por
modalidad) cuyas salidas $F_1, F_2$ se comparan mediante **correlación cruzada global**
en el dominio de Fourier. La magnitud del espectro de Fourier es invariante a
traslación; aplicando esa magnitud a coordenadas log-polares, rotación y escala se
convierten en traslaciones en ese dominio transformado, que también se resuelven por
correlación cruzada (ahora sobre $\log \rho, \theta$). La traslación final se resuelve
con una tercera correlación cruzada sobre las features ya rotadas/escaladas. Cada paso
usa un estimador de expectativa sobre un softmax de la superficie de correlación
(ecuación 11–12 de DPCN 2020), para que todo el pipeline sea diferenciable y el error de
pose final retropropague hasta los U-Net.

**La pregunta clave del usuario, respondida con precisión:** *"¿opera sin
correspondencias locales, así que un oráculo de keypoints no lo falsa?"* — **Cierto
en la letra, falso en el fondo.** DPCN no necesita **puntos discretos** emparejados, pero
la correlación cruzada — global o en el dominio espectral, da igual — es matemáticamente
una suma/producto de valores de $F_1$ y $F_2$ en posiciones correspondientes bajo la
pose candidata. Es literalmente **NCC generalizada y diferenciable**, evaluada de forma
eficiente vía FFT en vez de por fuerza bruta. Si en la pose GT el contenido de $F_1$ y
$F_2$ no covaría — que es exactamente lo que mide el test de información mutua del
oráculo (`analysis/measure_task2_oracle_signal.py`), y que dio **MI menor que el null en
casi todos los casos** para 9 proyecciones distintas —, no hay pico de correlación que
encontrar, sea local o global. El oráculo usa MI, un detector de dependencia no
paramétrica más general que NCC; si MI no encuentra señal en ninguna proyección clásica,
DPCN tampoco la va a encontrar operando sobre esas mismas intensidades sin transformar.

**Lo que sí deja abierto DPCN, y que el oráculo NO probó:** los U-Net de DPCN son
*aprendidos*, no proyecciones fijas. Podrían, en teoría, encontrar una transformación no
lineal del contenido donde sí exista correlación espectral aunque las 9 proyecciones
crudas no la tengan. Pero esto es exactamente lo que intentó CoMIR (par de encoders
aprendidos para maximizar correlación cruzada entre modalidades) y **ya fracasó**: el
encoder de OCT convergió a ignorar el volumen OCT (shuffle no degradaba la métrica). El
mecanismo de entrenamiento de DPCN — maximizar directamente el pico de correlación de
pose contra el GT — es más "on-task" que la pérdida contrastiva de CoMIR, así que no es
estrictamente el mismo experimento. Pero el riesgo de colapso a una solución degenerada
(features que producen *algún* pico, no necesariamente informativo, memorizando el prior
de pose en vez de leer la imagen) es el mismo tipo de fallo, y CoMIR ya demostró que
ocurre en este dataset concreto.

**Segundo hallazgo, independiente del anterior — y más contundente para descartar DPCN
tal cual en Task 2:** el paper DPCN++ documenta su propia limitación en la §6.4:

> *"[Fig...] reveals the fact DPCN++ might be degraded to the inputs with relatively
> small overlaps."*

El experimento médico real de DPCN++ (CT↔MRI↔ultrasonido) registra **la misma
estructura anatómica** (hueso) vista con dos contrastes distintos — es decir, dos vistas
con solapamiento casi total del mismo campo de visión, exactamente el régimen para el
que la correlación de fase (heredera directa de la correlación de fase clásica de
Reddy & Chatterji 1996, pensada para imágenes que comparten la mayor parte del
encuadre) fue diseñada. **Nuestro caso es lo contrario**: la proyección en-face del OCT
ocupa ~2.4% del área del fundus (memoria `fido-task2-scale-domain-shift`, escala
126–226 px sobre una imagen de 1024 px). Es un régimen de solapamiento pequeño casi por
definición — el mismo régimen que los propios autores señalan como el punto débil de su
método.

**Veredicto B: DPCN/DPCN++ no sobrevive a la premisa de "sin correspondencia de
apariencia".** No es una alternativa al matching de keypoints que evada el problema; es
una forma más eficiente de calcular lo mismo que NCC/MI ya midieron negativo, aplicada
además en un régimen de solapamiento pequeño que los propios autores documentan como
punto de fallo. Implementarlo exigiría: (1) resolver primero el problema de
"pequeño-dentro-de-grande" con un recorte/localización previo (como hace STHN/UASTHN,
ya identificado en ronda 1), y (2) aceptar el riesgo de colapso tipo CoMIR en la parte
de aprendizaje de features — ninguno de los dos requisitos está resuelto por leer el
paper con más detalle.

---

## Frente C — Registración sin correspondencia de apariencia

Buscado explícitamente: información mutua clásica, correlación de gradiente/bordes,
momentos de imagen, siluetas/contornos, priors geométricos de la distribución de poses.

**Información mutua (el clásico, Viola–Wells / Maes et al. 1995–97).** No están en
arXiv (pre-fecha el repositorio, journals cerrados) — no descargados, pero no hace
falta: el oráculo propio (`analysis/measure_task2_oracle_signal.py`) **ya calcula MI en
la posición GT para 9 proyecciones y da resultado negativo** (MI en GT menor que el
null en casi todos los casos). MI es precisamente la cantidad que ese método clásico
maximiza. El oráculo es, de hecho, una implementación directa del test que MI-based
registration necesitaría pasar para funcionar, y no lo pasa.

**Gradientes/bordes (HOG, PIIFD, esquinas FAST, top-hat de vasos).** Encontrado
abundante trabajo — incluyendo el único paper localizado sobre exactamente nuestro par
de modalidades, **Niu et al. 2014** (`2014_niu_sd-oct-enface-color-fundus-patch-matching.pdf`,
SD-OCT en-face ↔ foto de fondo de ojo a color) — pero **todos, sin excepción, dependen
de vasos** como la característica compartida (top-hat multiescala para segmentarlos,
luego matching de puntos locales sobre ellos). El propio oráculo ya probó una de las 9
proyecciones como **gradiente** del volumen — el ingrediente base de HOG/PIIFD/FAST — y
dio la misma señal nula que el resto. Nota de contexto importante sobre Niu et al.: su
OCT es **diagnóstica de campo ancho** (SVPI cubre gran parte de la retina, vasos
prominentes), mientras la nuestra es **iOCT quirúrgica de huella pequeña** centrada en
el sitio de la cirugía, no necesariamente sobre vasculatura densa — una razón estructural
adicional (no solo la ya medida) de por qué el método no transfiere.

**Momentos de imagen / siluetas / contornos.** Encontrado el principio (momentos de
segundo orden para rotación, centroide para traslación) y un ejemplo reciente y
literal — **CARe** (`2025_li_care-cross-modal-fundus-registration-large-fov.pdf`),
que ataca exactamente nuestra geometría de "template pequeño dentro de imagen grande"
con un pipeline de recorte-y-alineación (RANSAC + ajuste polinómico) — pero **vuelve a
depender de vasos** (registra OCTA, que es angiografía, contra fondo de ojo de campo
ancho). Ningún método de silueta/contorno puro se encontró aplicado a
OCT↔fundus específicamente; el candidato más cercano en espíritu, "Silhouette-to-Contour
Registration" (arXiv 2511.14343, escaneo intraoral↔radiografía cefalométrica), registra
dos modalidades que comparten un **contorno anatómico duro y bien definido** (hueso/diente),
que no tiene análogo evidente en nuestro par (la proyección en-face de un volumen OCT no
tiene un borde anatómico nítido que se corresponda con un borde visible en el fundus;
el "borde" de la huella del OCT es un artefacto del recorte del escáner, no una
estructura anatómica). No se descargó por no ser aplicable de forma directa.

**Un candidato genuinamente no probado por el oráculo: MIND (Heinrich et al. 2012,
Medical Image Analysis, sin PDF en abierto — no descargado).** MIND no compara
intensidades ni gradientes crudos; compara **auto-similitud local** (cuánto se parece
cada parche a sus vecinos desplazados, dentro de la misma imagen), bajo la premisa de
que esa estructura de auto-similitud —no el valor de intensidad— se preserva entre
modalidades. Es una señal distinta de las 9 que el oráculo ya midió. **No se puede
afirmar que esté muerta** con la evidencia actual — sería la única vía de este frente
que merece un oráculo dedicado antes de descartarla (coste bajo: computar MIND sobre la
proyección en-face y sobre el fundus en la posición GT, medir NCC/MI del descriptor
igual que se hizo con intensidad, gradiente y densidad de vasos).

**Priors geométricos de la distribución de poses (predecir la media en vez de mirar la
imagen).** No se encontró un paper específico de este patrón aplicado a registración
retina/quirúrgica — pero la pregunta se puede responder directamente con estadística ya
medida (memoria `fido-task2-geometry`, sobre 1214 casos):
- **Rotación**: cubre el círculo completo −180° a 180°, sin concentración aparente. Un
  prior de rotación (predecir la media, o cualquier valor fijo) **no puede** batir a un
  modelo que mira la imagen — no hay distribución concentrada que explotar.
- **Traslación**: tx = 472 ± 125, ty = 467 ± 140 sobre una imagen de 1024 px. La
  dispersión (125–140 px) es un orden de magnitud mayor que el presupuesto de error del
  AUC (10 px). Un prior constante falla por el mismo motivo.
- **Escala**: aquí sí hay concentración — pero **por escenario**, no globalmente (sd
  1.8–4.2 px dentro de cada escenario, ~2%, contra 129–177 px de media entre escenarios).
  Un prior de escala solo funciona si se sabe a qué escenario pertenece el caso — que
  es información de zoom del microscopio, no de contenido de imagen — y en Mock Test la
  escala real (213–226) está fuera del rango de entrenamiento (126–188), así que ni
  siquiera un prior "por escenario" perfecto en train ayudaría en Mock sin extrapolar.

**Veredicto C: no se encontró ningún método publicado que registre OCT↔fundus, o pares
retinianos heterogéneos en general, sin apoyarse en alguna forma de correspondencia de
apariencia (vasos, gradiente, intensidad) — y todas las formas de apariencia que
probamos ya están muertas por el oráculo.** El único hueco genuino y no explorado es
MIND (auto-similitud local), y merece un oráculo barato antes de cualquier otra cosa. El
prior de distribución de poses **no sirve como sustituto de un modelo visual**: solo
la escala tiene concentración suficiente para ser útil, y precisamente la escala es el
eje que ya está capado por el corrimiento de dominio (frente D) — así que un prior de
escala por sí solo, sin generalizar fuera del rango de train, no resuelve nada nuevo que
la aumentación de escala ya prevista no resuelva mejor.

---

## Frente D — Cómo resuelve la literatura la escala fuera de dominio

**El problema en números** (memoria `fido-task2-scale-domain-shift`): train
[126.66, 188.09] px, Mock [213.81, 226.16] px, sin solape. Es **extrapolación real**, no
interpolación dentro de un rango cubierto por aumentación insuficiente.

**Hallazgo duro, contra la intuición de "meter más escalas arregla la generalización":**
la búsqueda general sobre extrapolación de escala en redes profundas confirma un
resultado bien establecido — entrenar con múltiples escalas **dentro** de un rango no
garantiza generalización **fuera** de ese rango; aprender que un objeto de tamaño 1 es
igual a tamaño 2 no implica que la red lo reconozca en tamaño 4. Esto no invalida la
aumentación de escala ya planeada (recortar+reescalar el fundus y actualizar la matriz
GT exactamente, memoria `fido-task2-scale-domain-shift`) — esa aumentación **sí cubre
directamente el rango de Mock** porque se construye para llegar hasta ahí, no es
extrapolación disfrazada de aumentación. Pero sí es una advertencia: si la aumentación
no alcanza a cubrir el rango real de Mock (o de la Final Round, con escenarios nuevos
aún más lejos), no hay que esperar que el modelo "generalice solo" más allá de lo que
vio.

**Alternativa arquitectónica real, no solo aumentación de datos — Jansson & Lindeberg
2022** (`2022_jansson_scale-invariant-scale-channel-networks.pdf`). Proponen redes con
**canales de escala foveados** (cada canal procesa una porción de imagen cada vez más
grande a resolución cada vez menor, con pesos compartidos entre canales y pooling
max/avg entre ellos) que, entrenadas con **una sola escala**, generalizan a un rango de
**8x** de escalas nunca vistas — precisamente extrapolación, no interpolación. El ratio
train/Mock de FIDO es 226/127 ≈ 1.8x, muy por debajo del 8x que demuestran cubrir sin
aumentación. Esto es relevante como **red de seguridad arquitectónica**: si la
aumentación exacta ya planeada no alcanza (p. ej. en la Final Round con escenarios aún
más lejos de train), una cabeza de predicción de escala con estructura de canales
foveados es una alternativa con evidencia de funcionar por diseño, no por haber visto el
rango en datos.

**Candidato secundario — SA-Homo** (`2026_xie_sa-homo-scale-adaptive-homography.pdf`,
2026, aún sin revisión por pares formal — arXiv reciente). Ataca variación de escala de
hasta 8x en estimación de homografía con atención multiescala + refinamiento iterativo.
No queda claro en el paper si el 8x incluye escalas fuera del rango de entrenamiento
(extrapolación real, como Jansson & Lindeberg) o si su conjunto de entrenamiento ya
cubre ese rango con aumentación (en cuyo caso es una arquitectura mejor para el caso ya
cubierto, no una solución al corrimiento de dominio). Se registra como referencia de
diseño (atención cross-scale), no como método verificado para extrapolación.

**Veredicto D: la aumentación exacta ya prevista (recorte+reescalado del fundus con
actualización exacta de la matriz GT) sigue siendo la apuesta correcta y no la
contradice nada de lo encontrado — construye directamente la cobertura del rango de
Mock en vez de depender de que el modelo extrapole.** Lo nuevo que aporta esta ronda es
una alternativa concreta si esa aumentación no basta o si aparecen escenarios aún más
lejos en la Final Round: una cabeza de predicción de escala con arquitectura de canales
foveados (Jansson & Lindeberg), que tiene evidencia publicada de extrapolar 8x sin haber
visto esas escalas — muy por encima del 1.8x que separa train de Mock.

---

## Resumen por frente

| Frente | ¿Sobrevive a "vasos no corresponden"? | Qué hacer |
|---|---|---|
| A — instrumento como landmark | El **principio**, sí (ya validado por nosotros: enriquecimiento 3.0–3.6x sobre 312 puntos). Ningún **método publicado** sirve tal cual (todos requieren cinemática o video). | Construir detector propio de instrumento en en-face (reusar segmentación de Task 1 proyectada) + detector en microscopio (Task 1 ya lo tiene) + Procrustes de 2 puntos. Cobertura parcial (no todos los casos tienen instrumento dentro de la huella del OCT). |
| B — DPCN/DPCN++ | **No.** Es correlación cruzada global — matemáticamente lo mismo que el oráculo de MI ya midió negativo — más un riesgo de colapso tipo CoMIR, más un régimen de solapamiento pequeño que los propios autores señalan como punto débil. | Descartar como apuesta principal. No implementar sin resolver antes localización gruesa (STHN/UASTHN) y aceptar el riesgo de colapso. |
| C — sin apariencia | **No, salvo un hueco.** MI clásica: ya falsada por el oráculo. Gradiente/bordes: ya falsada (proyección de gradiente del propio oráculo). Momentos/siluetas: sin literatura aplicable a este par, sin estructura de contorno compartida obvia. Prior de distribución de pose: solo la escala está concentrada, y es justo el eje capado por el corrimiento de dominio. | **MIND (auto-similitud local) es el único hueco no probado** — vale un oráculo barato antes de descartarlo. El resto, cerrado. |
| D — escala fuera de dominio | N/A (no es un problema de apariencia, es de cobertura del rango). | La aumentación exacta ya planeada sigue siendo correcta. Canales de escala foveados (Jansson & Lindeberg) como red de seguridad arquitectónica si la aumentación no alcanza. |

## Papers descargados esta ronda

6 PDFs nuevos en `papers/`, indexados en `papers/INDEX.md` § "Ronda 2":
`2018_zhou_hand-eye-calibration-oct-robotic-eye-surgery.pdf`,
`2016_vasconcelos_similarity-registration-ultrasound-calibration.pdf`,
`2014_niu_sd-oct-enface-color-fundus-patch-matching.pdf`,
`2025_li_care-cross-modal-fundus-registration-large-fov.pdf`,
`2022_jansson_scale-invariant-scale-channel-networks.pdf`,
`2026_xie_sa-homo-scale-adaptive-homography.pdf`. Todos verificados (`%PDF` + tamaño
>700 KB). Los dos papers de DPCN/DPCN++ ya estaban en el repo desde la ronda 1 y se
leyeron completos (extracción de texto) para este informe, sin necesidad de descargarlos
de nuevo.
