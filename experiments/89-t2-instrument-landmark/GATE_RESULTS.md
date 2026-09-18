# T2-89 — resultado del gate (Procrustes GT-a-GT sobre instrumento)

**Gate pre-registrado en `INFORME.md` S5.** Ejecutado 100% local (CPU, sin
pod, sin GPU): anotaciones de `data/_annotations/Task 2/` y segmentación del
volumen leída directamente de `data/Task 2/Scenario_NN.zip` vía `zipfile`
(sin extraer los 28 GB completos — solo las ~200 KB de `Segmentation/*.png`
por caso necesario). No hizo falta ningún insumo del pod.

**Veredicto: NO-GO, inequívoco.** Ambas condiciones de NO-GO se disparan por
uno o dos órdenes de magnitud, no por poco: AUC = 0.0055 (umbral NO-GO:
< 0.10) y error medio = 935 px (umbral NO-GO: > 40 px). No cae en zona gris.

---

## 0. Aviso obligatorio: esto es un techo, no un score esperado

Este gate usa la **matriz GT** para (a) proyectar los keypoints de fundus a
uv y decidir qué casos "cuentan" (los 127 con ≥2 puntos dentro de la huella)
y (b) como referencia para encontrar, en el lado OCT, el píxel segmentado
más cercano a esa proyección GT. Es decir: mide el **límite superior** de la
vía si tuviéramos acceso a la matriz GT — que es exactamente lo que no
tenemos en inferencia. Un solver real necesitaría además **detectar** el
instrumento en fundus y en el volumen OCT sin la matriz GT, y esos
detectores no existen (`INFORME.md` S4.2). Si el techo ya falla aquí, un
solver real solo puede ser igual de malo o peor.

---

## 1. Bug encontrado y corregido antes de reportar el número

Al implementar el ajuste de la similitud de 4 DOF se probó primero
reutilizar `fido.geometry.fit_closed_form_similarity` (Kabsch/SVD para
rotación propia) + `compose_similarity(reflect=True)`, tal como sugiere su
propio docstring ("el signo se aplica después... reconstruyendo desde los
parámetros recuperados aquí").

**Verificación antes de confiar en el número** (disciplina exigida por
`CONSTITUTION.md`): se construyó un caso sintético con una matriz reflejada
conocida `M` y 2 puntos exactos relacionados por esa `M`. La reconstrucción
vía `fit_closed_form_similarity` + `compose_similarity(reflect=True)` dio
**138 px de error** donde el error correcto es **0 px** (los 2 puntos son
exactos, sin ruido). Causa: con exactamente 2 puntos la matriz de covarianza
del ajuste SVD es de rango 1 — el signo de la rotación (propia vs. reflejada)
queda indeterminado por los datos; rotación propia y reflejada ajustan igual
de bien esos 2 puntos pero divergen en cualquier otro punto (como las 4
esquinas que puntúa la métrica).

**Corrección**: se escribió `fit_reflected_similarity` en
`analysis/run_task2_instrument_gate.py`, que impone la estructura fija
`A=[[a,b],[b,-a]]` (la misma que usa `compose_similarity(..., reflect=True)`
en `src/fido/geometry.py:26-52`, confirmada 100% del tiempo sobre los 1214
casos por T2-R1/R3) como restricción **lineal** del ajuste — evita la
ambigüedad de reflexión por construcción en vez de por suerte. Verificado
sobre el mismo caso sintético (2 y 3 puntos, sin ruido): error residual
`~1e-12 px`. El script y este chequeo quedan en
`analysis/run_task2_instrument_gate.py` (función `fit_reflected_similarity`,
con la verificación documentada en su docstring).

Se documenta esto explícitamente porque el número final cambió con la
corrección (el primer intento, con el bug, daba AUC=0.0000 exacto en el
subconjunto; el número corregido de abajo es el que vale) y el proyecto ya
se equivocó antes por no verificar antes de reportar.

---

## 2. Comando exacto y salida real

```
python analysis/run_task2_instrument_gate.py \
  --annotations-root "data/_annotations/Task 2" \
  --zips-root "data/Task 2" \
  --output-json "experiments/89-t2-instrument-landmark/gate_results.json"
```

Log completo: `experiments/89-t2-instrument-landmark/gate_run.log` (127
líneas, una por candidato, más el resumen). JSON completo, caso por caso:
`experiments/89-t2-instrument-landmark/gate_results.json`.

Resumen impreso por el script (íntegro):

```
casos totales: 1214
candidatos (>=2 puntos dentro de la huella): 127

candidatos OK (score calculado): 99/127
desglose de status: {'ok': 99, 'empty_instrument_mask': 28}

AUC sobre el subconjunto OK (n=99): 0.0055
error medio px (subconjunto OK): 935.28
error mediano px (subconjunto OK): 309.01

AUC extrapolado a los 1214 casos (0 en los que no llegan a n_ok): 0.0004

desglose por escenario (candidatos / OK / AUC-escenario):
  Scenario_01: total=152 candidatos=44 ok=27 AUC=0.0000 error_medio=528.24px
  Scenario_02: total=154 candidatos=0 ok=0 AUC=nan error_medio=nanpx
  Scenario_03: total=83 candidatos=0 ok=0 AUC=nan error_medio=nanpx
  Scenario_04: total=127 candidatos=33 ok=26 AUC=0.0000 error_medio=1997.63px
  Scenario_05: total=108 candidatos=0 ok=0 AUC=nan error_medio=nanpx
  Scenario_06: total=121 candidatos=0 ok=0 AUC=nan error_medio=nanpx
  Scenario_07: total=76 candidatos=10 ok=9 AUC=0.0606 error_medio=575.97px
  Scenario_08: total=145 candidatos=0 ok=0 AUC=nan error_medio=nanpx
  Scenario_09: total=183 candidatos=40 ok=37 AUC=0.0000 error_medio=573.20px
  Scenario_10: total=65 candidatos=0 ok=0 AUC=nan error_medio=nanpx
```

**28/127 candidatos (22%) no llegaron ni a tener un score**: el keypoint de
fundus proyectaba dentro de la huella uv, pero el volumen OCT de ese frame
no tiene **ningún** píxel segmentado como instrumento (clases 8/10/11) en
toda la proyección en-face (`status="empty_instrument_mask"`). Esto es
además del problema de precisión: en 22% de los "candidatos" no hay ni
siquiera un punto OCT que ofrecer.

---

## 3. Los tres números pedidos

| Métrica | Valor | Qué responde |
|---|---:|---|
| AUC sobre los 127 candidatos (empty_mask cuenta como fallo) | **0.0043** | "si funciona, funciona aquí" — versión estricta, cuenta los 28 sin score como fallo |
| AUC sobre los 99 casos con score calculado | **0.0055** | "si funciona, funciona aquí" — solo donde SÍ hubo píxel de instrumento que buscar |
| **AUC extrapolado a los 1214 casos** (0 en los que no tienen ≥2 puntos) | **0.0004** | techo real de esta vía sobre la métrica completa de Task 2 |

La sospecha del encargo ("ronda 0.10") **no se confirma — es más bajo**:
el techo real sobre los 1214 casos es **0.0004**, dos órdenes de magnitud
por debajo de 0.10. Interpretación del "0" en la extrapolación, para que
quede sin ambigüedad: a los 1214-99=1115 casos sin score calculado se les
asigna error=+infinito (nunca están a ≤t px para ningún t de 0 a 10) — es
decir, contribuyen 0 casos-acertados en cada umbral de la fórmula del AUC.
No se les asigna "0 px de error" (eso sería lo opuesto, y falso).

Adicional pedido: error mediano sobre el subconjunto con score, **309.01 px**
(el error medio, 935.28 px, está inflado por unos pocos casos catastróficos
— ver sección 6). De 99 casos con score, solo **1** tiene error ≤10 px
(el umbral más generoso que puntúa la métrica), y ese mismo es el único
≤15 px y ≤40 px también — es decir, no hay ningún caso en la "zona GO" ni
en la "zona gris": o está muy bien (1 caso, 4.43 px) o está muy mal (98
casos, 46 px a 34,770 px).

---

## 4. Aplicación de los umbrales pre-registrados, sin moverlos

De `INFORME.md` S5: **GO** si AUC-subconjunto ≥ 0.30 y error medio ≤ 15 px;
**NO-GO** si AUC-subconjunto < 0.10 o error medio > 40 px.

- AUC-subconjunto medido: **0.0055** (127-candidatos: 0.0043) — muy por
  debajo de 0.10. **Dispara NO-GO.**
- Error medio medido: **935.28 px** — muy por encima de 40 px. **Dispara
  NO-GO.**

Las dos condiciones de NO-GO se cumplen de forma independiente y por un
margen de más de un orden de magnitud cada una. No hace falta invocar la
zona gris: esto no es un caso límite, es un NO-GO limpio.

---

## 5. Comparación con los techos conocidos

| Referencia | AUC | Fuente |
|---|---:|---|
| Baseline actual en el leaderboard | 0.0000 | `NOW.md` |
| **Este gate — techo del instrumento (127→1214 extrapolado)** | **0.0004** | esta corrida |
| Este gate — techo del instrumento (solo 99 casos con score) | 0.0055 | esta corrida |
| Techo T2-80, matriz GT descompuesta y reensamblada perfecta ("TODO GT") | 0.6389 | `ATTACK_LADDER.md:567` |
| Líder del leaderboard | 0.475 | `RESULTS.md` / `NOW.md` |

El techo del instrumento (0.0004) es indistinguible del 0.00 actual y está
**tres órdenes de magnitud** por debajo del techo T2-80 (0.6389) que sirve
de referencia de "cuánto vale la pena perseguir un enfoque de 4 DOF". La vía
de instrumento, tal como está planteada (Procrustes desde 2-3 keypoints
anotados/segmentados), no se acerca ni de lejos a ese techo — ni siquiera en
su versión más favorable (con matriz GT, sin ruido de detección).

---

## 6. Por qué falla tan mal — diagnóstico, no solo el número

Dos causas, ambas verificadas, no solo sospechadas:

**(a) La correspondencia geométrica es floja**, como ya adelantaba el
sanity check de Mock Test en `INFORME.md` S4.3 (n=5, ahora confirmado con
n=99 en train): el punto de fundus proyectado por la matriz GT rara vez cae
cerca del píxel de instrumento realmente segmentado. Ejemplo inspeccionado a
mano (`Scenario_01/00001`): la máscara de instrumento ocupa una región
compacta de la proyección en-face (bbox y=[88,104], x=[356,433] sobre
128x512), pero los 3 keypoints de fundus proyectados caen en
(x=282,y=49), (296,61), (3,77) — 100 a 350 px de distancia en el eje nativo
de 512, muy fuera de esa región. Esto es consistente con la propia tabla del
oráculo original: al radio más laxo usado allí (r=0.04, ~4% del lado), solo
el **7.7%** de los puntos de instrumento caían dentro (`oracle_train.md`) —
es decir, incluso en el escenario más favorable, más del 92% de los puntos
NO tienen una correspondencia cercana.

**(b) Amplificación geométrica por línea base corta**: varios de los peores
casos (p.ej. `Scenario_04/00033`, error 34,770 px) tienen 2 de sus 3
keypoints de fundus a **2.07 px** de distancia entre sí (los puntos
"Right Head Tip" / "Left Head Tip" de la pinza, casi coincidentes en esa
pose). Un ajuste de 4 DOF con 2 puntos casi coincidentes es numéricamente
mal condicionado: un pequeño error en la correspondencia OCT de cualquiera
de esos 2 puntos se traduce en un ángulo/escala completamente distinto, que
luego se proyecta sobre las 4 esquinas del cuadrado unitario (que están
lejos del centroide de los puntos usados para el ajuste). Esto agrava (a),
no lo reemplaza — incluso los casos con separación razonable entre puntos
siguen muy por encima de 40 px.

---

## 7. Veredicto

**NO-GO.** El techo de la vía de instrumento (asumiendo matriz GT conocida,
sin ruido de detección) es **0.0004** sobre los 1214 casos de train —
indistinguible del 0.00 actual, tres órdenes de magnitud por debajo del
techo T2-80 (0.6389). No se recomienda construir los detectores de la
sección 4.2 de `INFORME.md` (segmentador de instrumento OCT, segundo
keypoint fundus): el cuello de botella no es la falta de detectores, es que
incluso con anotaciones/segmentación GT perfectas de ambos lados, la
correspondencia geométrica entre "dónde el fundus dice que está el
instrumento" y "dónde el volumen OCT lo tiene segmentado" no alcanza
remotamente la precisión de pocos-px que exige la métrica de Task 2. La vía
de instrumento como landmark de registración se cierra aquí.

Lo que sigue en pie de `INFORME.md`: el ratio de enriquecimiento del
oráculo original (T2-R11/R11b) sigue siendo real y sigue confirmando que la
convención de ejes es la correcta — eso no se invalida por este resultado.
Lo que se cierra es específicamente la idea de usar esos mismos puntos como
landmarks para un solver de 4 DOF: la señal es real pero demasiado imprecisa
para la tarea.

---

## Archivos de esta corrida

- `analysis/run_task2_instrument_gate.py` — script del gate (nuevo),
  incluye `fit_reflected_similarity` con la verificación sintética en su
  docstring.
- `experiments/89-t2-instrument-landmark/gate_run.log` — salida real
  completa de la corrida citada arriba.
- `experiments/89-t2-instrument-landmark/gate_results.json` — resultado caso
  por caso (status, n_points, corner_error_px) más los agregados.

## Lo que NO se hizo

- No se corrió nada sobre GPU ni pod: todo con las anotaciones y zips ya
  presentes en el disco local.
- No se ajustaron los umbrales de GO/NO-GO a posteriori — son los mismos de
  `INFORME.md` S5, escritos antes de correr el gate.
- No se investigó más allá de los dos casos inspeccionados a mano (S6); con
  un NO-GO tan claro (órdenes de magnitud, no un caso límite) no se
  justifica gastar más tiempo en diagnóstico fino de por qué falla cada caso
  individual.
