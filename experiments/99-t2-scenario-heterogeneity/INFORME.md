# T2-99 — Por qué el modelo geométrico de Task 2 funciona en unos escenarios y falla por completo en otros

**Fecha**: 2026-08-19. Script reproducible: `analysis/analyze_task2_scenario_heterogeneity.py`
(`KMP_DUPLICATE_LIB_OK=TRUE python analysis/analyze_task2_scenario_heterogeneity.py`;
todo numpy/scipy, local, sin GPU, ~15s sobre los 1214 casos de train). Resultados
crudos en `experiments/99-t2-scenario-heterogeneity/scenario_heterogeneity_results.json`.
Métrica oficial: `fido.geometry.corner_error`/`corner_auc`, réplica exacta de
`vendor/fido/Codabench Bundle/scoring_program/scoring_registration.py` (ya usada,
sin cambios, por `analysis/derive_task2_geometry.py`).

Parte de `experiments/98-t2-pose-estimation/GEOMETRIA_EXACTA.md` (modelo:
`theta = -roll_iOCT + theta0`; `scale`/`tx`/`ty` vía gnomónico de 1er orden de
`(tilt, azimuth)` respecto a un eje de mira `v0` calibrado). Ese informe cerraba
con el desglose por escenario (AUC 0.0–0.69) sin investigar la causa. Este
informe la investiga.

## Resultado headline

**No es el rango de tilt del iOCT (mi sospecha inicial) ni ningún parámetro
oculto constante-por-escenario en la anotación. Es que los coeficientes de
escala/traslación del modelo (`s0`, `c_tx`, `c_ty` — ajustados por mínimos
cuadrados sobre 9 escenarios agrupados) no transfieren al décimo escenario:
cuando se ajustan usando SOLO los datos del propio escenario de test (un
oráculo, no desplegable), el AUC LOSO pooled sube de 0.1512 a 0.7204 y CADA
escenario, incluidos los tres que puntúan 0.0000, sube por encima de 0.61.**
Aislado mediante ablaciones: el eje de mira (`v0`) casi no importa (oráculo de
solo-eje: 0.1512→0.1601); son los coeficientes de escala+traslación juntos
(0.1512→0.7204) — la rotación en pantalla (`theta0`) tampoco importa
(0.1512→0.1507). No se encontró qué variable de la anotación predice esos
coeficientes por escenario (la única candidata, `Eyeball.Rotation[z]`, no
sobrevive el chequeo de robustez). **Conclusión honesta**: la vía no es
calibrable en test porque las escenas de test son nuevas y sin ejemplos
etiquetados de Task 2 para ajustar esos coeficientes.

## 1. Reproducción del desglose por escenario (verificado, idéntico a GEOMETRIA_EXACTA.md §4)

| escenario | n | error medio (px) | mediana (px) | AUC |
|---|---:|---:|---:|---:|
| 07 |  76 |  2.92 |  2.57 | **0.6890** |
| 08 | 145 |  3.04 |  2.59 | **0.6796** |
| 03 |  83 |  9.22 |  8.62 | 0.2125 |
| 06 | 121 | 13.37 | 12.93 | 0.0616 |
| 04 | 127 | 13.56 | 12.20 | 0.0379 |
| 01 | 152 | 16.43 | 15.95 | 0.0102 |
| 09 | 183 | 18.33 | 18.42 | 0.0065 |
| 02 | 154 | 20.41 | 19.23 | 0.0000 |
| 05 | 108 | 44.23 | 41.47 | 0.0000 |
| 10 |  65 | 29.45 | 26.93 | 0.0000 |
| **pooled** | **1214** | **16.85** | **14.56** | **0.1512** |

## 2. Las cuatro hipótesis pedidas — con números

### 2a. Rango angular de tilt del iOCT — REFUTADA como explicación principal

Medí, por escenario, el ángulo `tilt` (desviación respecto al eje de mira `v0`
calibrado globalmente, la misma cantidad de GEOMETRIA_EXACTA.md §2.1):

| escenario | tilt medio | tilt [min, max] | AUC |
|---|---:|---|---:|
| 07 | 7.47° | [0.76°, 15.70°] | 0.6890 |
| 08 | 7.32° | [0.83°, 16.11°] | 0.6796 |
| 03 | 11.10° | [1.51°, 19.85°] | 0.2125 |
| 02 | **6.83°** (el más bajo de los 10) | [0.59°, 13.83°] | **0.0000** |
| 10 | 10.80° | [3.18°, 19.89°] | 0.0000 |

`corr(tilt_medio, AUC) = -0.403`, `corr(tilt_máx, AUC) = -0.318` — negativas
(consistente con la dirección de la sospecha) pero **débiles y contradichas
directamente por el escenario 02**: tiene el tilt medio MÁS BAJO de los 10
escenarios y aun así AUC = 0.0000. El escenario 03 tiene el tilt medio más
alto y saca AUC = 0.21 (ni el mejor ni el peor). Si el gnomónico de 1er orden
se rompiera por ángulo grande, el ranking por tilt debería predecir el ranking
por AUC de forma mucho más limpia. **Se descarta como causa principal** (puede
contribuir marginalmente al ruido, no al colapso a AUC≈0).

### 2b. Escala GT por escenario — REFUTADA

`corr(escala_media_GT, AUC) = -0.053` — esencialmente cero. El escenario 05
tiene la escala media más baja (128.90) y AUC=0.0; el escenario 10 tiene la
escala media más alta (177.64) y también AUC=0.0; los dos mejores escenarios
(07/08, escala ~159) están en el rango medio, no en un extremo. La escala GT
en sí no distingue escenarios buenos de malos.

### 2c. Residuo de la relación de rotación (`theta = -roll + theta0`) — insuficiente

`corr(residuo_medio, AUC) = -0.483` — la correlación más fuerte de las tres
hipótesis de imagen/ángulo, pero **no es la causa operativa**: el escenario 02
tiene el residuo MÁS BAJO de los 10 (0.55°) y AUC=0.0000; el escenario 05
tiene residuo 0.66° (mejor que la media) y también AUC=0.0000. Se confirmó
además por ablación directa (§3): sustituir `theta0` por el valor ajustado con
el propio escenario de test cambia el AUC pooled de 0.1512 a 0.1507 —
**ningún cambio**. La pieza de rotación, ya identificada como la más sólida en
GEOMETRIA_EXACTA.md, no es la que falla.

### 2d. Parámetro fijo del simulador que en realidad varía por escenario — REFUTADA, sin excepción

Se comprobaron los 24 componentes de los 6 campos de pose crudos
(`Opmi.Translation`, `Opmi.Rotation`, `iOCT.Translation`, `iOCT.Rotation`,
`Eyeball.Translation`, `Eyeball.Rotation`) agrupando por escenario: para cada
componente se calculó la dispersión DENTRO de cada escenario (`within_sd`, la
más grande de las 10) y la dispersión de las medias ENTRE escenarios
(`between_sd`), y su razón.

Top hallazgo (de 24 candidatos, ordenados por razón `between/within`):

| campo | sd global | sd máx. dentro de escenario | sd entre medias de escenario | razón entre/dentro |
|---|---:|---:|---:|---:|
| `Eyeball.Rotation[y]` | 0.0035 | 0.0043 | 0.0021 | **0.5** |
| `iOCT.Rotation[x]` | 0.0583 | 0.0729 | 0.0237 | 0.3 |
| `Eyeball.Rotation[x]` | 0.0297 | 0.0388 | 0.0123 | 0.3 |
| `Eyeball.Rotation[z]` | 0.0106 | 0.0145 | 0.0030 | 0.2 |
| `Opmi.Translation[*]`, `iOCT.Translation[*]` | 0.0000 | 0.0000 | 0.0000 | 0.0 |

**Ningún campo tiene razón > 1.** Es decir: en TODOS los campos, lo que varía
dentro de un mismo escenario (ruido normal de la muestra) es igual o mayor que
lo que varía la media entre escenarios distintos. No existe un campo anotado
que sea "casi constante dentro de cada escenario, pero con un valor distinto
entre escenarios" — el criterio que se pidió comprobar como "decisivo" no se
cumple para ningún campo de la anotación cruda. **Se descarta con evidencia
exhaustiva** (los 24 componentes, no una muestra).

## 3. El mecanismo real: los coeficientes de escala/traslación no transfieren entre escenarios

### 3.1 El eje de mira sí varía entre escenarios, pero corregirlo no arregla nada

El eje `v0` (dirección de mira calibrada, GEOMETRIA_EXACTA.md §1.3) ajustado
usando SOLO los datos de un escenario difiere del ajustado con los otros 9
hasta **6.97°** (escenario 09) — muy por encima del listón de <0.5° que
GEOMETRIA_EXACTA.md §5 estableció como necesario para no destruir el AUC.
Esto sugería inicialmente que el eje era la causa. Pero al aislarlo (ablación
"solo el eje viene del oráculo, los coeficientes siguen ajustados con los
otros 9 escenarios, ahora relativos a ese eje correcto"):

```
axis_only:  AUC 0.1512 -> 0.1601   (practicamente sin cambio)
```

**El eje no es la causa.** La reparametrización `(gx, gy)` es una función
suave del eje; usar un eje "correcto" pero coeficientes fijados con los otros
9 escenarios (que están ajustados alrededor de su propio eje incorrecto) sigue
sin generalizar.

### 3.2 Los coeficientes sí son la causa — y son casi todo el efecto

Ablación inversa: eje de mira EXACTAMENTE como en el LOSO real (ajustado con
los 9 escenarios de train, potencialmente "equivocado" para el escenario de
test), pero los coeficientes (`theta0`, `coef_scale`, `coef_tx`, `coef_ty`)
ajustados usando SOLO el escenario de test (oráculo):

```
all_coefficients_only:  AUC 0.1512 -> 0.7200   mean_err 16.85px -> 2.60px
```

Esto recupera prácticamente TODO el AUC del oráculo completo (que además
recalibra el eje: 0.7054). **El eje de mira es casi irrelevante; lo que no
generaliza son los coeficientes de escala y traslación.**

### 3.3 Descomposición: hace falta escala + tx + ty juntos, no basta con uno solo

| coeficiente(s) recalibrado(s) con oráculo del propio escenario | AUC pooled | mean_err |
|---|---:|---:|
| ninguno (LOSO base) | 0.1512 | 16.85px |
| `theta0` solo | 0.1507 | 16.78px |
| `scale` (s0,s1) solo | 0.2252 | 14.03px |
| `tx` solo | 0.1814 | 13.84px |
| `ty` solo | 0.1606 | 15.00px |
| `tx`+`ty` | 0.1939 | 11.39px |
| `scale`+`tx` | 0.3602 | 9.84px |
| `scale`+`ty` | 0.3619 | 9.16px |
| **`scale`+`tx`+`ty`** | **0.7204** | **2.59px** |
| oráculo completo (+ eje) | 0.7054 | 2.75px |

`theta0` no aporta nada. Cada coeficiente aislado (`scale`, `tx`, `ty`) sólo
recupera una fracción modesta del AUC. Solo corrigiendo los tres juntos
aparece el salto grande. Esto es sobre todo un efecto **geométrico mecánico**,
no necesariamente una interacción física profunda: las 4 esquinas proyectadas
están a ~500-900px del origen de la imagen; para que las 4 caigan dentro de la
ventana de 10px que puntúa la métrica, `scale`, `tx` y `ty` tienen que ser
correctos simultáneamente — corregir solo 1 ó 2 de los 3 deja un desplazamiento
residual de esquina que sigue sin entrar en la ventana.

### 3.4 Magnitud del sesgo por escenario

El sesgo (bias, no dispersión: media de `predicho - GT` para cada escenario,
bajo LOSO real) en escala es sistemático y grande:

| escenario | sesgo de escala (px) | sesgo de traslación \|tx,ty\| (px) | AUC |
|---|---:|---:|---:|
| 07 | +0.03 | 2.29 | 0.6890 |
| 08 | +0.06 | 2.03 | 0.6796 |
| 05 | **+33.49** | 11.45 | 0.0000 |
| 10 | **-18.21** | 24.74 | 0.0000 |
| 02 | **-17.96** | 3.61 | 0.0000 |

`corr(|sesgo de escala|, AUC) = -0.728`; `corr(|sesgo de traslación|, AUC) =
-0.433`. El sesgo de escala es la variable más predictiva de las medidas en
todo el informe, pero **no está causada por el desajuste del eje de mira**:
`corr(desajuste_eje, |sesgo_escala|) = +0.072` (nulo), mientras que
`corr(desajuste_eje, |sesgo_traslación|) = +0.605` (moderado — el eje sí
explica parte del sesgo de traslación, coherente con §3.1, pero el sesgo de
traslación pesa menos en el AUC que el de escala).

## 4. ¿Qué explica que los coeficientes de escala/traslación difieran por escenario?

Se buscó, sin éxito confirmado, una variable de la anotación que prediga el
intercepto de escala (`s0`) ajustado por escenario (rango real: 126.7 en el
escenario 05 hasta 171.8 en el escenario 02 — 45px de rango sobre una base de
~150px). El mejor candidato de 8 componentes de cuaternión probados fue la
media por escenario de `Eyeball.Rotation[z]`:

```
corr(media Eyeball.Rotation[z], s0_oraculo) = -0.714   p=0.020   (n=10 escenarios)
```

Parece prometedor, pero **no pasa el chequeo de robustez**: la dispersión
DENTRO de cada escenario de ese mismo componente (hasta 0.0145) es 5× mayor
que la dispersión de sus medias ENTRE escenarios (0.0030) — el mismo criterio
de la hipótesis 2d, que ya lo había descartado como campo "casi constante por
escenario" (razón 0.2, ver tabla en §2d). Con solo 10 escenarios y 8
componentes de cuaternión probados, un p=0.02 no sobrevive corrección por
comparaciones múltiples (Bonferroni exigiría p<0.006). **No se encontró una
variable de la anotación que explique de forma fiable el sesgo de
escala/traslación por escenario.**

Esto es consistente con el intento fallido de cadena física completa
documentado en GEOMETRIA_EXACTA.md §3 (rayo del iOCT → esfera → cámara
pinhole, que no convergió a un ajuste competitivo, con el radio de la esfera
empujado a su límite). Una explicación físicamente plausible —no verificada
aquí, fuera del alcance de datos numpy/scipy sobre anotaciones— es que el
simulador use una geometría de escena (p. ej. el "radio de ojo" o la forma de
la superficie retinal del asset 3D) que difiere entre los 10 escenarios y que
simplemente no está expuesta en ningún campo de `Numerical/<frame>.json`: no
es traslación ni rotación, es una propiedad de forma del render que no tiene
campo dedicado en la anotación.

## 5. Honestidad sobre la generalización a test

Esto es lo más importante para decidir si vale la pena seguir esta vía:

- **La calibración por escenario SÍ funciona** (§3.2: AUC 0.72 con solo
  recalibrar `scale`+`tx`+`ty`; §3 previo mostró que hacerlo con TODO el
  modelo llega a 0.70-0.79 en cada escenario individual). El modelo
  geométrico como FORMA FUNCIONAL es correcto — el error residual tras
  calibración local es de 1.8-5.2px, dentro de la ventana de 10px.
- **Pero esa calibración usa el `Ground Truth.Task 2` del propio escenario de
  test para ajustar los coeficientes** — es un oráculo de diagnóstico, no un
  modelo desplegable.
- **No se encontró ninguna variable de la anotación (§2d, exhaustivo sobre 24
  campos) ni ninguna combinación simple de pose (§4) que prediga esos
  coeficientes sin usar el GT.** El único candidato (`Eyeball.Rotation[z]`)
  no es robusto.
- Los escenarios de test son escenas NUEVAS, sin ejemplos etiquetados de
  Task 2 propios para calibrar. Como no hay una vía verificada para estimar
  `s0`/`c_tx`/`c_ty` de un escenario nuevo —ni desde la anotación, ni (no
  investigado aquí, requeriría trabajo con imágenes, fuera del alcance
  numpy/scipy de este informe) desde el fundus/volumen OCT—, **la
  recalibración por escenario no es viable en producción tal como está
  planteada**.
- Conclusión práctica: el AUC de 0.1512 bajo LOSO (no el 0.70 del oráculo)
  sigue siendo la estimación honesta del techo de esta vía mientras no se
  encuentre una forma de estimar esos coeficientes sin ver el GT del
  escenario. El AUC alto de los escenarios 07/08 no es una señal de que el
  modelo "ya funciona en general" — es una coincidencia de que esos dos
  escenarios comparten un régimen de escala/traslación cercano al promedio
  global (sesgo de escala ≈0 en la tabla de §3.4), no de que el modelo haya
  resuelto la generalización entre escenarios.

## 6. Qué se descartó, con números, y qué queda abierto

Descartado con evidencia directa:
- Rango de tilt del iOCT (§2a): correlación débil y contradicha por el
  escenario 02.
- Escala GT (§2b): correlación ~0.
- Residuo de la relación de rotación (§2c): ablación directa confirma que
  `theta0` no mueve el AUC (0.1512→0.1507).
- Parámetro fijo del simulador con valor distinto por escenario, en
  cualquiera de los 24 componentes de pose anotados (§2d): ningún campo tiene
  razón entre/dentro-escenario > 1.
- Desajuste del eje de mira como causa del sesgo de escala (§3.1, §3.4):
  correlación ~0 con el sesgo de escala (aunque sí correlaciona con el sesgo
  de traslación, que pesa menos en el AUC).

Queda abierto, no resuelto en este informe:
- La causa física de por qué `s0`/`c_tx`/`c_ty` difieren por escenario. La
  hipótesis más plausible —una propiedad de forma del render (p. ej. radio o
  curvatura del ojo) no expuesta en la anotación— es consistente con el
  fallo de convergencia de la cadena física completa en GEOMETRIA_EXACTA.md
  §3, pero no se verificó aquí.
- Si esa variable fuera estimable desde las imágenes (fundus u OCT) en vez de
  desde la anotación, la vía de calibración por escenario podría rescatarse
  parcialmente — pero eso es una investigación de visión, no de anotaciones,
  y queda fuera del alcance numpy/scipy de este informe.
