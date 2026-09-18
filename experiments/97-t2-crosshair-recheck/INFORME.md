# R2 revisitado con números — ¿es detectable el crosshair/rectángulo de escaneo del iOCT en `opmi_image`?

**Fecha**: 2026-08-19. **Estado**: LOGRADO — se sostiene el cierre previo
("NO hay marcador dibujado explotable"), pero ahora con números, y con dos
hallazgos laterales reales y cuantificados que no bastan para explicar el
leaderboard.

---

## 0. Resumen ejecutivo

El cierre previo de R2 (`ATTACK_LADDER.md`: *"LOGRADO — NO. No está renderizado
en la imagen. Descartado como atajo."*) no traía ni un número. Se repitió la
pregunta con rigor cuantitativo sobre **208 casos reales** (203 de
entrenamiento en 5 escenarios + 5 del Mock Test) usando 4 pruebas
independientes. Resultado:

1. **No hay crosshair ni rectángulo dibujado en el píxel.** Confirmado por
   inspección visual (20 casos, recortes con contraste realzado) y por
   detección de líneas (Hough): el exceso aparente de líneas rectas en la
   posición anotada desaparece (p=0.244) al controlar por la excentricidad
   radial de la imagen.
2. **La prueba fotométrica ingenua SÍ mostraba una "señal" enorme (Δ≈+49
   niveles de R, Cohen's d≈1.98, p≈1e-34) — y es casi enteramente un
   artefacto de viñeteado**, no un marcador. Las imágenes de fundus caen de
   brillo medio 92→0 entre r=0 y r=600px desde el centro óptico; el centro
   del crosshair está sesgado hacia el centro de la imagen (distancia media
   194.8±80.5px vs máximo posible ~724px). Al controlar por excentricidad
   (control con el mismo radio, ángulo aleatorio) el efecto se reduce ~3-4x
   (d≈0.30–0.65) pero **no desaparece del todo** — hay un sesgo composicional
   residual real, moderado y difuso (no localizado como un marcador).
3. **Hallazgo lateral real, pero insuficiente**: la punta del fórceps
   (`Keypoints/Forceps/Right+Left Head Tip`, visible en el propio
   `opmi_image` — dominio 2D, sin pasar por el volumen OCT) está
   significativamente más cerca del centro del crosshair que el azar
   (distancia media 74.5px vs 248.9px nulo por permutación, p≈0 sobre 2000
   shuffles), pero (a) sólo cubre 41% de los casos, (b) el error mediano es
   81px, y (c) **incluso con la posición GT perfecta** y usando sólo el
   ángulo/escala promedio del dataset, el techo de `corner_auc` es **0.031**
   — muy por debajo de 0.475. La posición no es el cuello de botella a esta
   resolución de AUC; rotación y escala lo son.

**Ningún frente de este peldaño explica el 0.475 del líder.** El cierre de
R2 queda justificado con números; no se abre un atajo nuevo de "detectar el
marcador". Se documentan dos leads menores (sesgo composicional residual +
posición del instrumento) para quien quiera exprimir señal auxiliar, no como
solución.

---

## 1. Contexto y qué se sospechaba

`ATTACK_LADDER.md` cerró R2 sin ningún número. Tres razones para dudar del
cierre (dadas en el encargo de este peldaño):

1. El simulador anota `Keypoints/iOCT Microscope Crosshair` con coordenadas
   exactas — sabe dónde está.
2. Los sistemas de iOCT reales sí dibujan el área de escaneo en el visor.
3. En el leaderboard, quien puntúa alto en Task 1 (keypoints) tiende a
   puntuar alto en Task 2 — posible indicio de que Task 2 se resuelve
   detectando *algo* con las mismas herramientas de keypoints.

## 2. Datos usados

| Set | Escenarios | Casos | Cómo se obtuvieron |
|---|---|---:|---|
| Train | Scenario_03 (extraído completo) + Scenario_01/05/08/10 (30 casos c/u, extraídos selectivamente de los zips sin descomprimir el resto) | 203 | `experiments/97-t2-crosshair-recheck/extract_cases.py` |
| Mock Test (OOD) | Scenario_12 | 5 | ya extraído en `data/Mock Test/Task 2/` |

Todo local, sin tocar `vendor/fido/`, sin pods. Los datos extraídos de los
zips viven en `experiments/97-t2-crosshair-recheck/cache/` (no se commitea:
regenerable con el script de extracción).

## 3. Metodología (4 pruebas)

Script: `analysis/measure_task2_crosshair_signal.py`.
`RNG_SEED=20260819` fijo — reproducible byte a byte.

### A. Fotometría en las líneas del crosshair vs control

Puntos muestreados por interpolación bilineal a lo largo de los dos brazos
anotados (`Start 0↔End 0`, `Start 1↔End 1`, saltando ±8% alrededor del cruce
central). Dos controles, **por caso** (delta pareado, no promedio bruto):

- **Ingenuo**: puntos uniformes en toda la imagen, a ≥25px de las líneas.
- **`matched` (excentricidad controlada)**: un punto de control por cada
  punto del crosshair, al **mismo radio respecto al centro óptico de la
  imagen** (±15px), ángulo aleatorio. Aísla la señal del crosshair del
  viñeteado.

Métricas: Δ por canal RGB, Δ contraste local (std en ventana 7×7), Δ magnitud
de borde (Sobel). Reporte: media pareada, Cohen's d, Wilcoxon signed-rank.

### B. Detección de líneas (Hough)

Canny (umbral adaptativo por mediana) + `HoughLinesP` sobre cada imagen.
"Hit" = algún segmento detectado coincide (distancia ≤8px, ángulo ≤10°) con
alguno de los dos brazos anotados. Nulo ingenuo: 20 traslaciones aleatorias
del crosshair dentro de la imagen. Nulo `matched`: 20 traslaciones al mismo
radio de excentricidad, ángulo aleatorio. Test de dos proporciones
(`statsmodels.proportions_ztest`).

### C. Estadísticos de la región escaneada vs región nula

`project_corners(matrix_GT)` da el paralelogramo exacto que barre el OCT.
Se compara contra: (a) 30 traslaciones nulas dentro de límites válidos de la
imagen (ingenuo), (b) 30 traslaciones nulas al mismo radio de excentricidad
respecto al centro óptico (`matched`). Métricas: brillo medio, contraste
(std), saturación media (HSV), ruido de alta frecuencia (residuo tras blur
gaussiano).

### D. Inspección visual

`analysis/render_task2_crosshair_recheck.py` → 20 casos × 4 renders
(`_full`, `_overlay`, `_crop` ×3, `_crop_stretch` con CLAHE) en
`experiments/97-t2-crosshair-recheck/renders/` (80 PNGs).

### Hallazgo lateral: posición del instrumento (no estaba en el plan original, surgió de la inspección visual)

`analysis/measure_task2_instrument_position_signal.py`: distancia entre los
keypoints 2D del instrumento en `opmi_image` (`Keypoints/Forceps`,
`Keypoints/Endoilluminator` — **no** los del volumen OCT; esto es distinto
del oráculo cross-modal ya descartado en T2-R11/T2-R11b) y el centro del
crosshair GT (`matrix[:2,2]`), con nulo por permutación (shuffle entre casos,
2000 repeticiones) y simulación de `corner_auc` usando esa posición +
ángulo/escala medios del dataset.

---

## 4. Resultados

### A. Fotometría — train (n=203)

| Métrica | Δ ingenuo (media) | d ingenuo | p ingenuo | Δ `matched` | d `matched` | p `matched` |
|---|---:|---:|---:|---:|---:|---:|
| R (0-255) | **+48.6** | 1.98 | 2.5e-34 | +12.6 | 0.65 | 6.0e-15 |
| G | +36.0 | 1.81 | 3.1e-34 | +6.5 | 0.41 | 1.9e-06 |
| B | +22.6 | 1.75 | 7.0e-34 | +2.9 | 0.30 | 2.9e-04 |
| contraste local | +1.26 | 1.32 | 3.4e-33 | +0.63 | 0.62 | 1.2e-15 |
| borde (Sobel) | +5.01 | 1.30 | 5.5e-32 | +2.07 | 0.54 | 6.3e-12 |

**Lectura**: el efecto ingenuo (d≈1.3–2.0, "enorme") colapsa 3-4x al
controlar por excentricidad, confirmando que es mayormente viñeteado. Queda
un residuo real y significativo (d≈0.3–0.65, "pequeño a moderado" en la
convención de Cohen) — la zona escaneada tiende a estar un poco más
iluminada/contrastada que un parche igualmente excéntrico, pero de forma
difusa, no como un trazo localizado.

Mock Test (n=5, subpotenciado): misma dirección en `matched` (R +18.5,
d=0.69; contraste +9.8, d=1.38) pero ningún p significativo — consistente
con train, no lo contradice.

### B. Hough — train (n=203)

| | hit rate GT | hit rate nulo | z | p |
|---|---:|---:|---:|---:|
| nulo ingenuo (traslación uniforme) | 5.91% | 2.14% | 3.48 | **5.0e-04** |
| nulo `matched` (excentricidad igual) | 5.91% | 4.21% | 1.17 | **0.244** |

**Lectura**: el "exceso de líneas rectas cerca del crosshair" también era en
parte un artefacto de que los bordes (vasos) se concentran hacia el centro
óptico. Controlado por excentricidad, la diferencia **no es significativa**
(p=0.244) — no hay evidencia de líneas rectas dibujadas en la posición
anotada. (El test pareado por caso con Wilcoxon sobre `hit − hit_rate_nulo`
da p muy bajo en ambas variantes porque compara una variable binaria contra
una tasa continua pequeña — no es la comparación correcta aquí; se reporta
por transparencia pero el test de dos proporciones es el que responde la
pregunta.)

Mock Test (n=5): 0/5 hits en la posición GT, sin señal (p=0.82).

### C. Región escaneada — train (n=203)

| Métrica | Δ ingenuo | d ingenuo | p ingenuo | Δ `matched` | d `matched` | p `matched` |
|---|---:|---:|---:|---:|---:|---:|
| brillo medio | +28.4 | 1.65 | 4.2e-34 | +7.4 | 0.58 | 7.0e-13 |
| contraste (std) | +5.1 | 0.67 | 3.6e-15 | +3.5 | 0.54 | 2.7e-11 |
| saturación media | −4.1 | −0.26 | 1.1e-04 | +8.3 | 0.65 | 3.2e-16 |
| ruido alta frecuencia | +0.78 | 0.89 | 5.5e-26 | +0.28 | 0.44 | 4.3e-07 |

**Lectura**: mismo patrón que A — el efecto ingenuo es en su mayoría
viñeteado; el residuo `matched` (d≈0.4–0.65) es real pero moderado y difuso.
Nota: la saturación cambia de signo al pasar del control ingenuo al
`matched` — el control ingenuo mezclaba muchos píxeles casi negros de la
periferia (saturación numéricamente inestable ahí), el `matched` es más
limpio y muestra que la región escaneada es, en promedio, algo más saturada.

### D. Inspección visual

20 casos revisados uno por uno (`renders/*_overlay.png`, `*_crop_stretch.png`).
**Ningún caso muestra un crosshair, rectángulo, banda de iluminación,
sombra ni borde renderizado en la posición anotada.** Las líneas rojas/cian
visibles en `*_overlay.png` y `*_crop.png` son **overlays dibujados por
nuestro propio script** (no están en `*_full.png`, que es el archivo crudo
como lo recibe `opmi_image`). En los recortes con contraste realzado
(CLAHE) tampoco aparece nada — sólo textura de retina, vasos, y en varios
casos el instrumento quirúrgico, que sí es un objeto real y visible del
simulador (ver sección siguiente).

### E. Hallazgo lateral — posición del instrumento (real, pero no alcanza)

| Candidato | cobertura | n | dist. media real | dist. media nula (shuffle) | p (perm., 2000) |
|---|---:|---:|---:|---:|---:|
| `forceps_midtip` (punto medio Right/Left Head Tip) | 40.9% | 83 | **74.5px** | 248.9px | **0.0** |
| `forceps_joint` (Joint Tip) | 40.9% | 83 | 130.3px | 270.9px | 0.0 |
| `forceps_centroid` | 40.9% | 83 | 84.1px | 251.4px | 0.0 |
| `endo_tip` (Endoilluminator) | 100% | 203 | 533.6px | 557.7px | 0.0 (pero la magnitud es irrelevante) |

`forceps_midtip` es el candidato útil: significativamente más cerca del
centro del crosshair que el azar, en 3 escenarios distintos de train.
`endo_tip` técnicamente "significativo" por el enorme n pero la distancia
media (534px) lo hace inservible como predictor.

**Techo de AUC con esta señal** (`compose_similarity` con ángulo/escala =
media circular del dataset, 158.6±15.0px escala / dispersión angular
≈27.7°):

| Esquema | n | error medio px | `corner_auc` |
|---|---:|---:|---:|
| posición GT perfecta + ángulo/escala medios | 203 | 57.9 | **0.0309** |
| posición del fórceps (real, imperfecta) + ángulo/escala medios | 203 (120 sin fórceps → usa `endo_tip`) | 377.1 | 0.0 |
| sólo `forceps_midtip` (83 casos con cobertura) + ángulo/escala medios | 83 | 95.6 | 0.0 |
| nulo (fórceps de OTRO caso, shuffle) + ángulo/escala medios | 203 | 455.0 | 0.0004 |

**Lectura decisiva**: incluso con la posición **perfecta** (GT) y usando
sólo el ángulo/escala promedio del dataset (sin ninguna predicción por
imagen), el techo de `corner_auc` es 0.031 — muy por debajo de 0.475 y
también por debajo del propio techo medido en T2-80 con modelo entrenado real
(0.048 con posición GT + resto del modelo). La posición NO es el cuello de
botella a la resolución de 10px que exige el AUC oficial; el ángulo y la
escala lo son. La señal del fórceps, aun siendo estadísticamente real, no
tiene ni la cobertura (41%) ni la precisión (81px mediana) para acercarse.

---

## 5. Interpretación pre-registrada, aplicada

> "Si hay CUALQUIER diferencia estadísticamente sólida entre la región
> escaneada (o el crosshair) y el control, es un frente nuevo y probablemente
> LA solución: dilo de inmediato y cuantifica cuán localizable es la región."
> "Si no hay nada tras los cuatro análisis, el cierre de R2 queda por fin
> justificado con números."

Lo que se encontró está en un punto intermedio, y hay que ser honesto en las
dos direcciones:

- **SÍ hay diferencias estadísticamente sólidas** (fotometría `matched`,
  región `matched`, posición del instrumento) que sobreviven controles
  serios y se repiten en 5 escenarios de train.
- **NO son "la solución"**: son efectos difusos (fotometría/región, d≈0.3–0.65,
  sin localización tipo keypoint) o cubren muy pocos casos con precisión
  insuficiente (instrumento: 41% cobertura, 81px mediana, techo de AUC 0.031
  incluso en el caso ideal). Ninguno se acerca a explicar 0.475.
- **El Hough test (líneas dibujadas) y la inspección visual SÍ dan negativo
  limpio** una vez controlado el confusor de excentricidad: no hay marcador
  ni línea recta renderizada en la posición del crosshair.

## 6. Veredicto sobre R2

**R2 se mantiene: NO hay un crosshair/rectángulo visualmente detectable como
atajo.** El cierre anterior ("LOGRADO — NO") era correcto, pero no
tenía números; ahora los tiene, incluida la prueba de que la primera versión
"ingenua" de este mismo experimento (fotometría/región sin controlar
excentricidad) *habría producido un falso positivo espectacular* (d≈1.3–2.0,
p≈1e-34) por viñeteado — una trampa real que cualquier auditoría futura de
"apariencia cerca de una posición anotada" en este dataset debe controlar por
radio desde el centro óptico antes de creer el resultado.

La correlación Task 1 ↔ Task 2 en el leaderboard **no se explica** por este
mecanismo (detectar un marcador o el instrumento no alcanza el AUC
observado); sigue siendo una pregunta abierta — plausiblemente ambos tasks
comparten dificultad de fondo (calidad de imagen, escenario) más que una
solución compartida vía el instrumento.

## 7. Si alguien quiere exprimir estos leads (no son prioritarios)

- El residuo fotométrico/de región `matched` (d≈0.5–0.65 en brillo, contraste
  y saturación) podría entrar como *feature auxiliar* de un regresor de
  traslación (no como detector de keypoint independiente) — su magnitud no
  justifica un peldaño dedicado dado el techo de 0.031 de AUC por posición
  sola.
- `forceps_midtip` podría usarse como **prior de traslación cuando está
  presente** (41% de los casos) para acelerar/estabilizar el entrenamiento de
  la cabeza de posición, pero no reemplaza un regresor entrenado — su cobertura
  parcial y su error de 81px lo hacen insuficiente como solución standalone.
- Ninguno de los dos amerita interrumpir el trabajo ya priorizado en
  `ATTACK_LADDER.md` (T2-R7 en adelante).

## 8. Reproducibilidad

```bash
python experiments/97-t2-crosshair-recheck/extract_cases.py
python analysis/measure_task2_crosshair_signal.py
python analysis/render_task2_crosshair_recheck.py
python analysis/measure_task2_instrument_position_signal.py
```

Artefactos:
- `experiments/97-t2-crosshair-recheck/results_train.json` /
  `results_mock.json` — reporte completo + filas por caso.
- `experiments/97-t2-crosshair-recheck/renders/` — 80 PNGs (20 casos × 4).
- `experiments/97-t2-crosshair-recheck/instrument_signal_log.txt` — log
  completo del hallazgo lateral.
- `experiments/97-t2-crosshair-recheck/cache/` — subconjunto de imágenes/JSON
  extraído de los zips (regenerable, no se commitea).
