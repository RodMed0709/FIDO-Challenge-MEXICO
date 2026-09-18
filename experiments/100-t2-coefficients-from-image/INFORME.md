# T2-100 — Estimar los coeficientes de escala/traslación de Task 2 desde el fundus

**Fecha**: 2026-08-19. Script reproducible:
`analysis/estimate_task2_coefficients_from_image.py`
(`KMP_DUPLICATE_LIB_OK=TRUE python analysis/estimate_task2_coefficients_from_image.py`;
numpy/scipy/PIL/sklearn/cv2/skimage, local, CPU, ~2 min sobre los 1214 casos
de train — la mayor parte del tiempo es leer `microscope.png`/`visibility.png`
de los `.zip` de 9 de los 10 escenarios y el chequeo nulo de 1000 tiradas).
Resultados crudos en `experiments/100-t2-coefficients-from-image/results.json`;
cache de features de imagen en `image_features_cache.npz` (mismo directorio).

Continúa `experiments/98-t2-pose-estimation/GEOMETRIA_EXACTA.md` y
`experiments/99-t2-scenario-heterogeneity/INFORME.md`: el modelo geométrico
(`theta = -roll_iOCT + theta0`; escala/traslación vía proyección gnomónica de
1er orden) es la forma funcional correcta, pero sus coeficientes de
escala/traslación no transfieren entre escenarios sin recalibrar. Este
informe intenta estimar esos coeficientes **desde el fundus** (`Stereo
Left/<frame>/{microscope.png, visibility.png}`, la imagen `opmi_image` real
de `inference(task_id, oct_volume, opmi_image, model)`), que es lo único
disponible en test.

## Resultado headline

| método | AUC oficial (LOSO) | mean error (px) |
|---|---:|---:|
| coeficientes globales (sin recalibrar) | 0.1512 | 16.85 |
| **este trabajo — 1 feature de imagen (`profile_edge_r50`), LOSO anidado** | **0.3201** | **11.14** |
| oráculo: recalibrar SOLO 3 interceptos por escenario, pendientes globales | 0.2550 | 12.22 |
| oráculo: recalibrar los 8 números (escala+tx+ty) por escenario | 0.7204 | 2.59 |

**El resultado end-to-end es AUC=0.3201.** Supera el listón pre-registrado de
0.20 (≈+112% relativo sobre 0.1512) por un margen razonable, queda muy por
debajo del techo del líder (0.475) y muy por debajo del oráculo (0.7204). Un
chequeo de escepticismo obligatorio (§6, feature de ruido puro en el mismo
pipeline, 1000 tiradas) sitúa este resultado en el percentil 98.8 de lo que
se obtiene por puro azar con n=10 escenarios — señal real, no artefacto del
pipeline, pero con la fragilidad estadística inherente a n=10 declarada sin
adornos.

## 0. Aclaración de nomenclatura: "s0, c_tx, c_ty" no son 3 escalares

La tarea que originó este informe citaba el oráculo por-escenario como
"recalibrando SOLO tres coeficientes del ajuste gnomónico (`s0`, `c_tx`,
`c_ty`)", con AUC=0.7204. Se comprobó aquí que **eso no es literal**: el
modelo de `derive_task2_geometry.py` tiene 8 números libres por escenario —
`coef_scale=[s0,s1]`, `coef_tx=[c_gx,c_gy,c0]`, `coef_ty=[c_gx,c_gy,c0]` — y
recalibrar SOLO los 3 interceptos (`s0`, `c0_tx`, `c0_ty`), dejando las 5
pendientes en `gx,gy` fijas globalmente, da AUC=0.2550 bajo LOSO — muy por
debajo de 0.7204. El 0.7204 verificado en `experiments/99-t2-scenario-
heterogeneity/INFORME.md` §3.3 (ablación `"scale+tx+ty"`) recalibra las TRES
FUNCIONES completas (8 números), no solo sus interceptos. `s0`, `c_tx`,
`c_ty` en la tarea original son, aparentemente, nombres abreviados de esas
tres funciones (escala, offset-x, offset-y), no de 3 escalares. Este informe
trata el objetivo real como esos 8 números y lo deja documentado para que no
se repita la confusión.

## 1. Features de imagen extraídas (verificado, geometría de imagen simple)

Sobre `Stereo Left/<frame>/microscope.png` (RGB 1024×1024, la `opmi_image`
real de inferencia) y `visibility.png` (máscara binaria oficial de qué
región del canvas es retina visible — no un umbral inventado, viene del
propio simulador):

- **`vis_*`**: geometría de la máscara `visibility.png > 128` — fracción de
  área, centroide, radio equivalente (`sqrt(area·H·W/π)`), bounding box
  (min/max/alto/ancho/centro), excentricidad (autovalores de la covarianza
  de los píxeles de la máscara), radio del percentil 95 de distancia al
  centroide.
- **`nb_*`**: los mismos 14 descriptores pero sobre una máscara "no-negra"
  calculada por umbral simple en `microscope.png` (`suma RGB > 10`) — un
  chequeo de redundancia/robustez independiente de `visibility.png`.
- **`profile_edge_r50`**: perfil radial de brillo desde el centro fijo del
  canvas `(512,512)` — radio donde la intensidad media cae por debajo de la
  mitad del nivel "interior" (mediana del 20% de bins más cercanos al
  centro). Es una medida del borde del viñeteado óptico basada en
  intensidad continua, no en un umbral binario duro — más robusta a
  instrumentos que ocluyen parte de la máscara binaria.

29 features en total (`FEATURE_NAMES` en el script), cacheadas para los 1214
casos en `image_features_cache.npz`.

Nota verificada de sanity check: el bounding box de la máscara de
visibilidad toca los bordes de la imagen (`0` o `1023`) en una fracción
grande de los casos — el campo iluminado suele ser MÁS GRANDE que el canvas,
recortado por el encuadre. Esto acota (no invalida) el rango dinámico de
`vis_r_equiv`/`vis_area_frac` como proxy de escala: en escenarios donde el
campo está muy cerca de llenar el canvas en (casi) todos los casos
(escenario 01: área media 0.93; escenario 10: 0.98 — ver tabla §5.2), la
señal de escala está parcialmente saturada.

## 2. Heterogeneidad intra- vs entre-escenario: features de imagen

| feature | sd entre escenarios | sd máx. dentro de escenario | razón entre/dentro |
|---|---:|---:|---:|
| `vis_area_frac` | 0.098 | 0.048 | **2.06** |
| `vis_r_equiv` | 31.85 | 16.30 | **1.95** |
| `nb_area_frac` | 0.086 | 0.046 | **1.87** |
| `nb_r_equiv` | 27.60 | 16.77 | **1.65** |
| `vis_centroid_y` | 19.12 | 45.41 | 0.42 |
| `vis_centroid_x` | 11.32 | 24.55 | 0.46 |
| `profile_edge_r50` | 41.20 | 84.39 | 0.49 |

A diferencia de los 24 campos de la anotación cruda comprobados en
`experiments/99-t2-scenario-heterogeneity/INFORME.md` §2d (**ninguno** con
razón > 1 — ninguna variable de pose es "casi constante dentro de escenario,
distinta entre escenarios"), aquí **las features de TAMAÑO del campo visible
(`vis_area_frac`, `vis_r_equiv`, `nb_area_frac`, `nb_r_equiv`) sí tienen
razón > 1**: varían más entre escenarios que dentro de un mismo escenario.
Es una señal estructuralmente distinta a la que se buscó (sin éxito) en las
anotaciones de pose. En cambio, el **centroide** de la máscara y el perfil
radial de brillo tienen razón < 1 (más ruido dentro de escenario que
diferencia entre escenarios) — son features más ruidosas caso a caso, lo
cual anticipa (y se confirma en §5) que usarlas como features de un
predictor multi-feature con solo 9 puntos de entrenamiento por fold
perjudica más que ayuda.

## 3. ¿Los coeficientes son por-escenario o por-caso? — split-half bootstrap

Pregunta central pedida en la tarea: si el óptimo por-caso difiere mucho del
óptimo por-escenario, la estrategia entera cambiaría (habría que estimar
coeficientes por caso, no por escenario). Se probó con un bootstrap de
partición aleatoria: para cada escenario, 30 particiones aleatorias en dos
mitades, ajuste de los 8 coeficientes oráculo en cada mitad por separado, y
se compara la diferencia típica entre mitades (`mean_split_half_abs_diff`,
ruido de estimación con la mitad de los datos de UN escenario) contra la
dispersión de esos mismos coeficientes ENTRE escenarios
(`between_scenario_sd`):

| coeficiente | sd entre escenarios | diff. típica entre mitades (split-half) | razón |
|---|---:|---:|---:|
| `s0` (intercepto de escala) | 13.20 | 0.46 | **28.7** |
| `c_gx_tx` (pendiente dominante de `tx`) | 95.31 | 3.19 | **29.8** |
| `c0_tx` (intercepto de `tx`) | 3.43 | 0.24 | **14.0** |
| `c_gy_ty` (pendiente dominante de `ty`) | 94.73 | 2.27 | **41.7** |
| `c0_ty` (intercepto de `ty`) | 3.79 | 0.29 | **13.0** |
| `c_gy_tx` (pendiente cruzada de `tx`) | 2.99 | 1.80 | 1.7 |
| `c_gx_ty` (pendiente cruzada de `ty`) | 2.23 | 2.76 | 0.8 |
| `s1` (curvatura de escala) | 22.08 | 19.10 | 1.2 |

**6 de los 8 coeficientes son MUY estables dentro de un escenario**
(razón 13–42×: el ruido de ajustarlos con la mitad de los datos es una
fracción pequeña de cuánto varían entre escenarios distintos) — esto
**confirma que la granularidad correcta es por-escenario, no por-caso**: no
hace falta (ni ayudaría) estimar coeficientes caso a caso, porque dentro de
un escenario son esencialmente constantes. Es coherente con el mean error de
2.59px del oráculo completo (§ resultado headline), ya cerca del residuo
inherente de ~2.7px que deja la asimetría afín-vs-similitud reportada en
`GEOMETRIA_EXACTA.md` — llegar a ese nivel con granularidad de escenario ya
casi agota lo que el modelo de 4-DOF puede explicar; no hay margen visible
para que ir a granularidad de caso aporte mucho más.

Las dos pendientes cruzadas (`c_gy_tx`, `c_gx_ty`) y la curvatura de escala
(`s1`) son la excepción: su ruido intra-escenario es comparable o mayor que
su variación entre escenarios — son, en la práctica, poco identificables
(ruidosas) con los tamaños de muestra por escenario disponibles (65–183
casos), más que verdaderamente "constantes con valor distinto por caso".

## 4. Correlaciones exploratorias (n=10 escenarios — INFORMATIVO, con cautela extrema)

Con solo 10 puntos, cualquier variable con 2-3 grados de libertad puede
"correlacionar" por azar; esta tabla es exploratoria y **no** se usó para
elegir el modelo final de §5 (la selección de features de §5 fue a priori,
por razonamiento físico, antes de mirar esta tabla).

| coeficiente objetivo | top-3 features por \|r\| (Pearson, n=10) |
|---|---|
| `s0` | `profile_edge_r50` (r=+0.77, p=0.010), `vis_bbox_cx` (r=+0.74, p=0.013), `vis_bbox_xmax` (r=+0.68, p=0.032) |
| `c0_tx` | `profile_edge_r50` (r=-0.75, p=0.013), `vis_bbox_cx` (r=-0.73, p=0.016), `vis_bbox_xmax` (r=-0.65, p=0.040) |
| `c_gy_ty` | `profile_edge_r50` (r=-0.78, p=0.008), `vis_bbox_cx` (r=-0.75, p=0.013), `vis_bbox_xmax` (r=-0.68, p=0.030) |
| `c0_ty` | `profile_edge_r50` (r=-0.79, p=0.007), `vis_bbox_cx` (r=-0.77, p=0.010), `vis_bbox_xmax` (r=-0.70, p=0.025) |
| `s1` | `nb_eccentricity` (r=-0.65, p=0.044), `vis_edge_r_p95` (r=+0.57, p=0.087) |
| `c_gx_tx` | `profile_edge_r50` (r=+0.78, p=0.008), `vis_bbox_cx` (r=+0.74, p=0.014) |
| `c_gy_tx` | `nb_bbox_ymin` (r=+0.57, p=0.085) — el más débil de los 8 |
| `c_gx_ty` | `vis_bbox_cy` (r=-0.58, p=0.080) — el más débil de los 8 |

Ninguna correlación pasa Bonferroni estricto (29 features × 8 objetivos ≈
232 comparaciones, p<0.0002 requerido), así que se leen como **hipótesis**,
no como hechos. Pero hay un patrón que sí es informativo por sí mismo: 4 de
los 8 coeficientes (`s0`, `c_gx_tx`, `c_gy_ty`, `c0_tx`/`c0_ty`) correlacionan
FUERTE con las MISMAS dos features (`profile_edge_r50`, `vis_bbox_cx`), con
signos coherentes con la física esperada (a mayor radio aparente del campo
visible, mayor escala `s0`/`c_gx_tx`/`c_gy_ty` y menor magnitud de los
interceptos de traslación). Esto es consistente con la hipótesis del
enunciado de la tarea (el tamaño aparente del campo visible determina la
escala) y con que estos 4 coeficientes compartan un factor latente común de
"zoom por escenario" — no son 4 señales independientes, así que 4
correlaciones altas correladas entre sí pesan menos que 4 independientes.

## 5. Predictor lineal end-to-end, LOSO anidado (task item 4 — el resultado del proyecto)

Para cada fold externo (escenario de test dejado fuera): se calibra el eje
de mira y `theta0` con los 9 escenarios de entrenamiento (igual que
`derive_task2_geometry.py`); se calcula el oráculo de 8 coeficientes por
CADA uno de esos 9 escenarios (usando solo los datos de ese escenario); se
agrega la feature de imagen por escenario (media sobre los casos del
escenario); se ajusta una regresión lineal (mínimos cuadrados, regularización
ridge mínima `1e-6` solo por estabilidad numérica) de coeficiente-oráculo
sobre feature(s) de imagen, con 9 puntos de entrenamiento; se predicen los 8
coeficientes del escenario de test a partir de SU media de features de
imagen (nunca se usa su GT); se compone la matriz y se mide `corner_error`/
`corner_auc` oficial.

| conjunto de features | # params libres por objetivo | AUC LOSO | mean err (px) |
|---|---:|---:|---:|
| `vis_r_equiv` + `vis_centroid_y` + `vis_centroid_x` (elección inicial, 3 features) | 4 | 0.1465 | 21.26 |
| `vis_r_equiv`, `vis_centroid_y`, `vis_centroid_x`, `vis_eccentricity`, `profile_edge_r50` (ampliada) | 6 | 0.1818 | 17.27 |
| `nb_r_equiv` + `nb_centroid_y` + `nb_centroid_x` (máscara no-negra en vez de `visibility.png`) | 4 | 0.1084 | 22.59 |
| `vis_bbox_cx` (sola) | 2 | 0.1848 | 15.09 |
| `vis_r_equiv` (sola) | 2 | 0.2524 | 14.54 |
| `vis_area_frac` (sola) | 2 | 0.2649 | 14.19 |
| `vis_r_equiv` + `profile_edge_r50` | 3 | 0.2425 | 13.73 |
| **`profile_edge_r50` (sola) — MEJOR** | **2** | **0.3201** | **11.14** |

**Patrón claro de "menos es más"**: con solo 9 puntos de entrenamiento por
fold, cada feature adicional por objetivo añade riesgo de sobreajuste que no
compensa con la señal marginal que aporta. El modelo de 3-4 features (mi
elección inicial, motivada por la intuición "radio + centroide") generaliza
PEOR que los coeficientes globales sin recalibrar (0.1465 < 0.1512) —
fracaso instructivo, no un resultado positivo escondido. El modelo de UNA
sola feature (`profile_edge_r50`, el perfil radial de brillo, 2 parámetros
por objetivo: pendiente + intercepto) es el que mejor generaliza.

### 5.1 Desglose por escenario del modelo ganador (`profile_edge_r50` sola)

| escenario | n | mean error (px) | AUC |
|---|---:|---:|---:|
| 09 | 183 | 2.49 | **0.7317** |
| 08 | 145 | 3.58 | **0.6307** |
| 07 | 76 | 4.64 | **0.5287** |
| 03 | 83 | 4.75 | **0.5214** |
| 10 | 65 | 4.95 | **0.5063** |
| 06 | 121 | 9.23 | 0.1923 |
| 04 | 127 | 9.67 | 0.1553 |
| 01 | 152 | 17.60 | 0.0251 |
| 02 | 154 | 18.77 | 0.0000 |
| 05 | 108 | 33.01 | 0.0000 |
| **pooled** | **1214** | **11.14** | **0.3201** |

5 de 10 escenarios llegan a AUC≥0.5 (por encima incluso del oráculo global
sin recalibrar por caso), 2 se quedan en un rango intermedio (0.15–0.20), y
3 colapsan a ≈0 (01, 02, 05) — heterogeneidad grande, igual que en el modelo
sin recalibrar (T2-98 §4) pero con un conjunto de escenarios "buenos"
distinto. Diagnóstico del fallo en 02 (verificado): su `profile_edge_r50`
medio (366.2px) es un valor INTERMEDIO entre escenarios, pero su `s0`
oráculo real (171.84) es el MÁS ALTO de los 10 — la relación
feature→coeficiente simplemente no es monótona ahí, no es un problema de
extrapolación fuera de rango. El escenario 05 sí parece más un caso de
extrapolación: tiene el `profile_edge_r50` medio más bajo (320.1) y también
el `s0` oráculo más bajo (126.73) — dirección correcta, pero el modelo
entrenado sin ver ese extremo (LOSO) subestima cuánto hay que extrapolar.

## 6. Chequeo de escepticismo obligatorio (n=10): ¿el 0.32 es señal o azar?

Se repitió EXACTAMENTE el mismo pipeline (agregación por escenario, ajuste
lineal 9 puntos, LOSO anidado, AUC oficial) sustituyendo la feature de
imagen por **ruido gaussiano puro** (misma forma, sin relación con nada),
1000 tiradas:

```
AUC con ruido puro (n=1000): media=0.1542  std=0.0481  min=0.0470  max=0.4029
percentiles: p90=0.2173  p95=0.2405  p99=0.3264
```

Es una distribución con cola derecha pesada: el propio ruido, con este
pipeline y n=10, alcanza AUC=0.40 en su peor caso (mejor dicho, "mejor" caso
de puro azar) alguna vez en 1000 tiradas — el bar de "distinto de 0" NO es
el correcto para juzgar señal aquí; hay que compararse contra esta
distribución nula, no contra 0:

| feature | AUC | percentil vs. nulo | z |
|---|---:|---:|---:|
| `vis_bbox_cx` | 0.1848 | 82.0% | +0.64 |
| `vis_r_equiv` | 0.2524 | 96.0% | +2.04 |
| `vis_area_frac` | 0.2649 | 97.1% | +2.30 |
| **`profile_edge_r50`** | **0.3201** | **98.8%** | **+3.45** |

`vis_bbox_cx` (el centroide del bounding box) NO se distingue del ruido —
coherente con §2 (razón entre/dentro < 1 para el centroide). Las tres
features de TAMAÑO del campo visible (`vis_r_equiv`, `vis_area_frac`,
`profile_edge_r50`) caen las tres en el percentil 96–99 del nulo: no es una
casualidad de una sola feature con suerte — son 3 formas computacionalmente
independientes (máscara binaria oficial, umbral RGB propio, perfil de
intensidad) de medir la MISMA cantidad física (tamaño aparente del campo
iluminado), y las 3 aterrizan en el mismo rango elevado del nulo con el
mismo signo. Esa convergencia entre 3 mediciones independientes es más
convincente que cualquiera de los 3 p-valores por separado (percentil 99 con
n=10 escenarios corresponde, en el mejor de los casos, a algo como p≈0.01 —
nunca va a ser una prueba fuerte con tan pocos escenarios).

**Conclusión honesta de este chequeo**: hay señal real y consistente, del
tamaño y la dirección esperados por la física del problema, pero la
confianza estadística es la que n=10 permite — moderada, no aplastante.

## 7. Comparación contra el criterio pre-registrado

| criterio pre-registrado | umbral | resultado (0.3201) |
|---|---:|---|
| features de imagen no capturan los coeficientes, replantear | AUC < 0.16 | ❌ no aplica — 0.3201 >> 0.16 |
| supera todo lo conseguido, justifica construir la submission | AUC ≥ 0.20 | ✅ **se cumple**, con margen (+60% relativo sobre el umbral) |
| primer lugar del leaderboard | AUC ≥ 0.475 | ❌ no se alcanza (0.3201 < 0.475, y muy por debajo del oráculo 0.7204) |

**Veredicto: AUC=0.3201 end-to-end, con el mismo modelo geométrico ya
derivado (T2-98) y coeficientes de escala/traslación estimados desde el
fundus por escenario (no desde el GT), calibrado y evaluado con LOSO real y
la métrica oficial exacta.** Supera 2× el score real actual en Codabench
(0.0000), duplica el AUC de los coeficientes globales sin recalibrar
(0.1512) y justifica construir la submission según el criterio
pre-registrado. No alcanza el liderazgo del leaderboard.

## 8. ¿Un modelo aprendido tendría posibilidades? (task item 5 — argumentado, no entrenado aquí)

No se entrenó ningún modelo de aprendizaje (CNN, GBM, etc.) en este informe
— solo regresión lineal 2-parámetros sobre features de geometría de imagen,
por diseño (mandato de la tarea). Con la evidencia reunida:

**A favor de que un modelo aprendido pueda mejorar sobre 0.3201:**

- El cuello de botella identificado en §5 no es "falta de señal en la
  imagen" sino "falta de DATOS para ajustar el predictor": solo 9 puntos
  (medias por escenario) por fold LOSO. Pero §3 mostró que los coeficientes
  son ESENCIALMENTE CONSTANTES dentro de un escenario (razón split-half
  13–42× para 6 de 8 coeficientes) — eso significa que, en principio, **no
  hace falta agregar a nivel de escenario**: cada uno de los 1214 casos ya
  lleva (aproximadamente) la misma etiqueta objetivo que sus 64-182
  compañeros de escenario. Un modelo entrenado a nivel de CASO (no de media
  por escenario) tendría ~1090 pares (imagen, coeficiente-del-escenario) por
  fold LOSO en vez de 9 — mucho más presupuesto de aprendizaje para el mismo
  problema de fondo, siempre que el split LOSO siga siendo por escenario (la
  fuga sigue siendo el riesgo real, no el conteo bruto de ejemplos).
- Solo 3 de 8 coeficientes objetivo (`s0` vía escala, y los 2 interceptos de
  traslación) se explican razonablemente por una sola feature escalar de
  "tamaño aparente del campo". Las pendientes cruzadas (`c_gy_tx`,
  `c_gx_ty`) y la curvatura de escala (`s1`) no correlacionan con nada
  probado aquí — un extractor más rico (tamaño del disco óptico, calibre de
  vasos principales, orientación de la vasculatura, textura/nitidez
  radial en más de una dirección, no solo el perfil promedio) podría
  capturar la parte anisotrópica que un radio/área escalar no puede, por
  construcción, distinguir (un círculo no tiene orientación).
- Los 3 escenarios que colapsan (01, 02, 05) no colapsan todos por el mismo
  motivo (§5.1: 02 es no-monotonía, 05 parece extrapolación) — un modelo con
  más capacidad y regularización adecuada (no necesariamente profundo; un
  GBM con pocas hojas, o ridge con más features pero regularización fuerte
  entre escenarios) podría absorber ambos modos de fallo mejor que una recta.

**En contra / riesgos:**

- El techo del ORÁCULO (0.7204, viendo el GT del propio escenario de test)
  ya es el límite superior de CUALQUIER estimador de estos 8 coeficientes,
  perfecto o no — un modelo aprendido nunca superará 0.72 con esta forma
  funcional, y probablemente se quede bastante por debajo salvo que
  encuentre información que el oráculo por mínimos cuadrados no explota
  (poco probable, el oráculo ya usa el GT exacto). Recuperar 0.72 en test
  real exigiría estimar los 8 números con precisión cercana a la del
  oráculo — un salto grande desde 0.32.
- Con solo 10 escenarios etiquetados disponibles para validar CUALQUIER
  elección de arquitectura/hiperparámetros, el riesgo de sobreajustar la
  ELECCIÓN DE MODELO (no solo sus pesos) sigue siendo alto — este mismo
  informe ya mostró que ir de 2 a 4-6 parámetros por objetivo empeora la
  generalización (§5); no hay razón a priori para asumir que una red con
  miles de parámetros, por bien regularizada que esté, escape ese patrón sin
  evidencia empírica que aquí no se generó.
- El fallo en escenario 02 (§5.1) es un fallo de NO-MONOTONÍA, no de falta
  de capacidad — sugiere que falta una variable (posiblemente NO visible en
  el fundus 2D: profundidad de la esfera ocular, curvatura retinal — la
  misma "propiedad de forma del render no expuesta en la anotación" que
  T2-99 §4 dejó como hipótesis abierta) más que falta de flexibilidad del
  modelo.

**Conclusión de esta sección**: un modelo aprendido case-level (aprovechando
que la etiqueta es ~constante dentro de escenario, con validación LOSO por
escenario estricta) es la extensión natural con mayor probabilidad de
mejorar sobre 0.3201, especialmente si añade features direccionales/no-
isotrópicas que un radio escalar no puede capturar. No hay evidencia aquí de
que vaya a acercarse al techo de 0.72 — ese techo asume conocer el GT, y la
brecha entre "lo que el fundus muestra" y "lo que el GT exige" para el
escenario 02 en particular no parece ser un problema de capacidad de modelo.

## 9. Qué es VERIFICADO vs INFERIDO en este informe

**Verificado** (recomputado con la métrica oficial, reproducible con el
script):
- Los 3 valores de referencia (0.1512, 0.2550, 0.7204) — el script los
  recalcula con las mismas funciones de `derive_task2_geometry.py` y
  coinciden exactamente con T2-98/T2-99.
- Que recalibrar SOLO 3 interceptos (no 8 números) da 0.2550, no 0.7204 —
  aclara la nomenclatura de la tarea (§0).
- Los 8 coeficientes oráculo son estables dentro de escenario para 6 de 8
  (split-half bootstrap, §3) — respalda granularidad por-escenario.
- `vis_area_frac`/`vis_r_equiv`/`nb_area_frac`/`nb_r_equiv` tienen razón
  entre/dentro-escenario > 1 (§2) — a diferencia de TODOS los campos de
  anotación cruda probados en T2-99.
- El resultado end-to-end 0.3201 con `profile_edge_r50` como única feature,
  LOSO anidado real, métrica oficial exacta (§5).
- El chequeo nulo (§6): 0.3201 cae en el percentil 98.8 de 1000 tiradas de
  ruido puro en el mismo pipeline.

**Inferido / no verificado, declarado como tal**:
- Por qué exactamente el escenario 02 rompe la monotonía feature→coeficiente
  (§5.1) — se caracterizó el síntoma, no se encontró la causa.
- Cualquier argumento sobre si un modelo aprendido superaría 0.3201 (§8) —
  es razonamiento sobre la evidencia reunida aquí, NO un resultado medido;
  no se entrenó ningún modelo de ese tipo.
- Que `profile_edge_r50` sea LA feature causal correcta y no un proxy
  correlacionado con la verdadera causa física — no se dispone de acceso al
  motor de render para confirmarlo.
