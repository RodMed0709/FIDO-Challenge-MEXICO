# T2-98 — Derivación de la composición geométrica exacta

**Fecha**: 2026-08-19. Script reproducible: `analysis/derive_task2_geometry.py`
(ejecutar `python analysis/derive_task2_geometry.py`; todo numpy/scipy, local,
sin GPU, ~10s sobre los 1214 casos de train). Resultados crudos en
`experiments/98-t2-pose-estimation/derive_task2_geometry_results.json`.

Continúa el hallazgo de `HALLAZGO.md`: Task 2 es estimación de pose, no
registración de imágenes. Este informe intenta derivar la fórmula geométrica
EXACTA (no una regresión polinómica genérica) que compone la matriz GT a
partir de las poses, y la verifica con la métrica oficial.

## Resultado headline

| método | AUC oficial (LOSO) | parámetros libres |
|---|---:|---:|
| baseline lineal (18 componentes cuaternión → 6 params matriz) | 0.0937 | ~108 |
| **este modelo (geometría + gnomónico de 1er orden)** | **0.1512** | **~10** |
| techo: GT reproyectado como similitud reflejada de 4 DOF | 0.6389 | — |
| techo: matriz GT exacta (AUC=1 por definición) | 1.0000 | — |

Mejora real (+61% relativo sobre el baseline lineal) con **un orden de
magnitud menos parámetros**, usando **solo `iOCT.Rotation`** (no
`Eyeball.Rotation`, ni traslaciones, todas constantes). Pero el modelo está
lejos del techo de 0.6389: la parte de **rotación de la matriz (`theta`) se
deriva de forma casi exacta**; la parte de **escala y traslación es una
aproximación de primer orden con residuo real (~7-17 px)**, no la fórmula
exacta. Ver honestidad de la sección "Qué queda sin resolver".

## 1. Qué es VERIFICADO (a precisión numérica o casi)

### 1.1 La matriz GT es exactamente afín

Sobre los 1214 casos, la fila 3 de `Ground Truth.Task 2` es `[0,0,1]` con
desviación `0.0` (precisión de máquina), en el 100% de los casos. No hay
componente proyectivo: es un mapeo afín del cuadrado unidad a píxeles, nunca
una homografía con perspectiva real.

### 1.2 Solo `iOCT.Rotation` importa (Eyeball.Rotation es prescindible)

Regresión cuadrática de los 6 parámetros de la matriz sobre cuaterniones,
en muestra (n=1214):

| features | R² por parámetro |
|---|---|
| `iOCT.Rotation` solo | 0.968, 0.967, 0.992, 0.967, 0.968, 0.993 |
| `Eyeball.Rotation` solo | 0.023, 0.016, 0.164, 0.019, 0.023, 0.370 |
| ambos | 0.976, 0.975, 0.994, 0.975, 0.976, 0.994 |

`iOCT.Rotation` solo ya explica prácticamente todo. Esto es consistente con
el argumento físico: el ojo es una esfera (radio fijo, centro en el origen
del mundo, `Eyeball.Translation` = `[0,0,0]` en el 100% de los casos); el
*lugar geométrico* de puntos que ocupa una esfera es invariante a rotaciones
sobre su propio centro. El rayo del escáner iOCT intersecta ese lugar
geométrico en un punto que **no depende de cómo esté orientado el ojo**, solo
de hacia dónde apunta el escáner. La proyección de ese punto a través de la
cámara del microscopio (Opmi, pose 100% fija) tampoco depende de la rotación
del ojo. El pequeño resto de R² que aporta `Eyeball.Rotation` (0.16–0.37 en
2 de los 6 parámetros) se investigó (§3) y **no mejora el AUC oficial bajo
LOSO** — es sobreajuste, no señal física real explotable con el modelo actual.

### 1.3 El eje de mira local del iOCT es el eje local `-Y`

`Opmi.Spatial.Rotation` es fijo y conocido: `[0.70711, 0, 0, 0.70711]`
(cuaternión xyzw) = rotación de +90° en torno al eje X mundial. Verificado:
esta rotación lleva el eje local `Z` del Opmi a `(0,-1,0)` mundial — el Opmi
mira hacia abajo (`-Y`), consistente con estar por encima del ojo
(`Opmi.Translation.Y = 96.7 > 0 = Eyeball.Translation.Y`).

Para el iOCT, buscando entre los 6 ejes locales candidatos (±X,±Y,±Z) cuál,
rotado por `iOCT.Rotation` caso a caso, apunta más consistentemente hacia el
origen del mundo (donde está el ojo): el eje local **`-Y`** es el que gana
por un margen aplastante — ángulo medio respecto a la dirección al origen
= **8.7°** (máximo 21.2°), frente a >85° para cualquier otro eje candidato.
Este es el eje de mira (`FWD_LOCAL_IOCT`) usado en el modelo. Nótese que el
iOCT usa `-Y` como mira mientras que el Opmi usa `+Z` — convención de ejes
locales distinta entre los dos objetos del motor de simulación (esperable,
son mallas/objetos distintos).

### 1.4 La rotación en pantalla (`theta`) es -roll del iOCT, casi exacto

Descomponiendo el cuaternión del iOCT en (`tilt`, `azimuth`, `roll`)
respecto al eje de mira `v0` calibrado en 1.3 (`tilt`/`azimuth`: desviación
angular de la mira respecto a `v0`; `roll`: rotación del eje local "derecha"
del iOCT alrededor de su propia mira, medida contra una referencia
transportada desde `v0`), y descomponiendo la matriz GT en
`(theta, scale, tx, ty)` vía `a = s·cosθ, b = s·sinθ` (fila 0 de la matriz,
igual que `fido.geometry.decompose_similarity`):

```
theta_GT = -roll_iOCT + theta0        theta0 = 0.06° (~0, sin ajustar)
```

Residuo angular sobre los 1214 casos: **media 0.78°, máximo 5.6°** — una
relación casi exactamente lineal (el ajuste de `[cosθ0, sinθ0]` da un vector
de magnitud 0.9998, donde 1.0 es una relación angular perfecta). Esta es la
pieza más sólida de la derivación: la orientación del recuadro escaneado en
la imagen es, literalmente, el negativo del roll del escáner alrededor de su
propio eje de mira, sin ninguna contribución medible de `Eyeball.Rotation`.

## 2. Qué es INFERIDO (funciona, pero es una aproximación, no la fórmula exacta)

### 2.1 Escala y traslación: gnomónico de primer orden

Con `gx = tan(tilt)·cos(azimuth)`, `gy = tan(tilt)·sin(azimuth)` (la
proyección gnomónica estándar de una dirección 3D sobre un plano tangente en
`v0` — el modelo de "cámara pinhole" de primer orden):

```
scale ≈ s0 + s1·(gx² + gy²)        s0=156.97  s1=93.22
tx    ≈ c_gx·gx + c_gy·gy + c0     c_gx=1104.36  c_gy=7.83   c0=471.94
ty    ≈ c_gx·gx + c_gy·gy + c0     c_gx=-16.21   c_gy=-1106.69 c0=467.25
```

(coeficientes ajustados sobre los 1214 casos de train, valores exactos en
`derive_task2_geometry_results.json`). Este es un ajuste de mínimos
cuadrados en forma cerrada (lineal en los coeficientes, sin optimización
iterativa) — 6 números libres en total.

Residuo (en muestra, sin LOSO): `tx` media 7.4px/máx 50.6px, `ty` media
8.3px/máx 60.7px, `scale` media 10.8px/máx 33.7px. **No es un ajuste
preciso** — deja error real dentro de la ventana 0-10px que puntúa la
métrica, y sube el error medio de esquinas a ~15-17px bajo LOSO.

Se probó enriquecer el modelo (términos cuadráticos/cúbicos en `gx,gy`,
proyección estereográfica en vez de gnomónica) sin mejora apreciable — el
residuo no baja de ~7px en traslación con ningún término polinómico
adicional de `gx,gy` solo. Esto sugiere que **falta un ingrediente físico
real** en el modelo (ver §3), no que falte flexibilidad polinómica.

## 3. Intento fallido: cadena física completa (rayo → esfera → cámara pinhole)

Se intentó el modelo completo pedido en la tarea: rayo del iOCT (origen fijo
`iOCT.Translation`, dirección `R_iOCT · d_local(u,v)` con `d_local` una
mezcla afín de `Forward/Right/Up` locales parametrizada por el field-of-view
del escaneo) → intersección con esfera de radio `R_eye` ajustable centrada en
el origen → proyección pinhole por la cámara Opmi (pose fija conocida,
intrínsecos `f, cx, cy` ajustables). 6 parámetros libres continuos
(`R_eye, su, sv, f, cx, cy`), resueltos por mínimos cuadrados no lineales
(`scipy.optimize.least_squares`, método `trf` con bounds) sobre las 4
esquinas proyectadas de los 1214 casos, con arranques múltiples (240
combinaciones de valores iniciales × permutación/signo de ejes locales
Right/Up).

**No convergió a un ajuste competitivo**: el mejor resultado encontrado tiene
error medio de esquinas ~49px en muestra (peor que el modelo gnomónico de
primer orden de §2, que da ~15px). El radio de la esfera se empuja
sistemáticamente hacia el límite superior permitido (`R_eye → |iOCT.Translation|`,
es decir, la cámara casi tocando la esfera) — señal de que el modelo está mal
especificado o subdeterminado, no de que haya convergido a un óptimo físico
real. Causas verosímiles no resueltas: (a) el patrón de escaneo del iOCT
podría no ser un raster rectangular simple de rayos divergentes desde un
único origen puntual (podría ser telecéntrico, con orígenes de rayo que
varían con `(u,v)` en vez de la dirección), (b) el "radio de esfera" con
`Eyeball.Translation=[0,0,0]` fijo podría no representar directamente la
superficie retinal que ve el iOCT, o (c) hay más de una superficie/etapa de
proyección involucrada que este modelo de un solo rebote no captura.

**Esto se deja documentado como negativo**: no se encontró la cadena física
de ray-tracing exacta dentro del esfuerzo de esta tarea. El modelo de §2
(gnomónico de primer orden, sin ray-tracing real) es el que mejor combina
precisión y honestidad de lo verificado.

## 4. Verificación con la métrica oficial (corner-AUC, leave-one-scenario-out)

`fido.geometry.corner_error` / `corner_auc` — réplica exacta de
`vendor/fido/Codabench Bundle/scoring_program/scoring_registration.py`.
Split: 10 folds, cada uno deja fuera un escenario completo (`Scenario_01`
… `Scenario_10`); la calibración del eje de mira (`v0,e1,e2`, §1.3) y los
coeficientes de §1.4/§2.1 se reajustan en cada fold usando solo los 9
escenarios de entrenamiento — sin fuga del escenario de test.

| escenario | n | error medio (px) | mediana (px) | AUC |
|---|---:|---:|---:|---:|
| 01 | 152 | 16.43 | 15.95 | 0.0102 |
| 02 | 154 | 20.41 | 19.23 | 0.0000 |
| 03 |  83 |  9.22 |  8.62 | 0.2125 |
| 04 | 127 | 13.56 | 12.20 | 0.0379 |
| 05 | 108 | 44.23 | 41.47 | 0.0000 |
| 06 | 121 | 13.37 | 12.93 | 0.0616 |
| 07 |  76 |  2.92 |  2.57 | 0.6890 |
| 08 | 145 |  3.04 |  2.59 | 0.6796 |
| 09 | 183 | 18.33 | 18.42 | 0.0065 |
| 10 |  65 | 29.45 | 26.93 | 0.0000 |
| **pooled** | **1214** | **16.85** | **14.56** | **0.1512** |

Heterogeneidad grande entre escenarios (AUC 0.0 en 02/05/10, AUC ~0.68 en
07/08) — indica que el `tilt` cubierto por cada escenario, o algún otro
factor específico de escenario, hace que la aproximación gnomónica de primer
orden funcione muy bien en algunos rangos angulares y muy mal en otros. No
investigado a fondo por límite de tiempo; queda como pista para refinar el
modelo de §2/§3.

Ablaciones (todas bajo LOSO, ver `analysis/derive_task2_geometry.py` +
scripts exploratorios): añadir `Eyeball.Rotation` aplanada (9 números) a la
regresión de traslación **empeora** el AUC pooled (0.096 con términos
lineales de `gx,gy`, 0.086 con cuadráticos, frente a 0.151/0.150 sin ojo) —
sobreajuste, no señal real explotable con este modelo. Términos cuadráticos
en `gx,gy` sin `Eyeball.Rotation` no cambian el resultado (0.150 vs 0.152).

## 5. Listón de precisión angular (barrido de sensibilidad)

Perturbando `iOCT.Rotation` con ruido angular de magnitud creciente (eje
aleatorio uniforme, 8 tiradas por magnitud) y recomponiendo la matriz vía el
modelo ajustado sobre todo train (§1.4+§2.1), midiendo el AUC oficial contra
la matriz GT real (sin perturbar):

| ángulo de ruido | AUC (media ± std, 8 tiradas) |
|---:|---:|
| 0.00° (sin ruido, referencia en muestra) | 0.1683 ± 0.0000 |
| 0.10° | 0.1620 ± 0.0022 |
| 0.25° | 0.1354 ± 0.0011 |
| 0.50° | 0.0726 ± 0.0048 |
| 1.00° | 0.0212 ± 0.0010 |
| 2.00° | 0.0034 ± 0.0009 |
| 5.00° | 0.0000 ± 0.0001 |

**El listón es extremadamente exigente**: con solo 0.5° de error angular en
la estimación de `iOCT.Rotation`, el AUC ya cae por debajo del baseline
lineal actual (0.0937); con 1° cae a 0.02, prácticamente inútil. Para
igualar el AUC en-muestra de este modelo (0.168) hace falta una precisión
angular **mejor que 0.1°** — sub-décima de grado. Esto es un requisito de
precisión angular extremadamente alto para cualquier estimador entrenado
sobre imagen (fundus/OCT); da una medida concreta de cuán difícil es la vía
"estimar pose desde imagen y componer" tal como está planteada hoy.

Perturbar `Eyeball.Rotation` en solitario (`perturb=eye_only` en el barrido)
da AUC constante = 0.1683 en toda la tabla, **por construcción**: el modelo
final no usa `Eyeball.Rotation` en absoluto (§1.2). Esto no significa que el
verdadero generador del simulador sea insensible a la rotación del ojo — solo
que, dentro de la evidencia y el esfuerzo de esta derivación, no se encontró
una forma de explotarla que generalizara bajo LOSO. Es una limitación
declarada del modelo, no una afirmación física fuerte sobre el simulador.

## 6. Qué queda sin resolver (honestidad)

- La cadena física exacta (rayo 3D → esfera → cámara pinhole) **no se logró
  ajustar** de forma competitiva (§3): es un negativo documentado, no un
  éxito parcial disfrazado.
- El modelo final (§1+§2) mejora el baseline lineal bajo LOSO (0.151 vs
  0.094) con muchos menos parámetros, pero está lejos del techo de 0.6389
  del propio espacio de matrices "similitud reflejada de 4 DOF" — la parte
  de traslación/escala tiene error real de ~15-17px, no ~0.
- La contribución de `Eyeball.Rotation` a la traslación (R² marginal
  0.16-0.37 en dos parámetros, §1.2) es real en el sentido estadístico pero
  no se logró convertir en mejora de AUC bajo LOSO — puede ser una señal
  física genuina y pequeña que este modelo no logra aislar del ruido, o
  puede ser una correlación espuria del diseño del dataset. No resuelto.
- La fuerte heterogeneidad entre escenarios (§4) no se investigó: podría
  deberse a que distintos escenarios cubren distintos rangos de `tilt` (el
  gnomónico de primer orden degrada con `tilt` grande) o a algún otro factor
  de escenario no modelado.

## 7. Conclusión para la escalera de ataque

La vía "estimar pose y componer geométricamente" tiene una pieza sólida
(la rotación en pantalla se deriva de `iOCT.Rotation` con error angular
casi nulo) pero la pieza de posición/escala no está resuelta a precisión
suficiente, y el listón de precisión angular necesario (§5, <0.1-0.5°) es
muy exigente para cualquier estimador entrenado sobre imagen. El AUC de
0.151 bajo LOSO con matrices GT es una COTA SUPERIOR optimista de lo que
esta vía podría lograr en producción (donde `iOCT.Rotation` tendría que
estimarse desde el volumen OCT, con error >> 0°) — y ya está lejos del líder
del leaderboard (0.475). Antes de invertir en un estimador de pose desde
imagen, valdría la pena: (a) resolver la cadena física exacta de §3 (el
modelo actual deja ~15px de error sin explicar, que probablemente sí tiene
una causa geométrica encontrable), o (b) evaluar si el presupuesto de
esfuerzo restante rinde más en otra rama de la escalera.
