# T2-95 — ¿Cuánto puntúa en Task 2 predecir SIEMPRE una matriz constante?

**Pregunta:** sin mirar ninguna imagen, ¿qué AUC oficial consigue una submission
de Task 2 que predice la misma matriz de homografía (similitud reflejada 4-DOF)
para todos los casos?

**Motivación:** el leaderboard tiene 7 equipos resolviendo Task 2 (cnielsen
0.475, Alex 0.376, oli4more 0.28, kwonmj 0.279, dancies 0.244…) mientras
nosotros estamos en 0.00. Cinco auditorías internas (cadena de medición ya
verificada limpia) coinciden en que no hay correspondencia visual explotable
entre iOCT y fundus (IoU de vasculatura 0.015–0.07 contra máscaras GT
perfectas). Hipótesis a probar: parte de ese 0.475 podría no venir de "ver"
nada, sino de que las poses estén lo bastante concentradas por escenario como
para que una constante bien elegida ya puntúe.

**Veredicto corto: la hipótesis muere.** El mejor caso posible (oráculo
por escenario, evaluado dentro del propio escenario) da **AUC = 0.0001**.
Leave-one-scenario-out (la medida honesta) da **AUC = 0.0000**. Sobre el Mock
Test da **AUC = 0.0000**. Todo muy por debajo del umbral de 0.05 que mataría
la hipótesis. El 0.475 del líder exige señal real que todavía no hemos
encontrado.

---

## 1. Método

### 1.1 Métrica — verificada contra el scorer oficial vendorizado

`vendor/fido/Codabench Bundle/scoring_program/scoring_registration.py`
proyecta las 4 esquinas del cuadrado unitario canónico por la matriz predicha
y por la GT, calcula el error L2 medio de las 4 esquinas por caso, y agrega
como AUC (accuracy media a umbrales enteros 0..10 px, `MAX_THRESHOLD_PX=10`).

Usamos `src/fido/geometry.py` (`project_corners`, `corner_error`,
`corner_auc`), que ya existía en el repo como réplica de ese scorer. Antes de
medir nada se verificó bit a bit contra el módulo vendorizado, cargándolo
directamente por `importlib` y comparando 200 pares de matrices aleatorias:

```
max abs diff per-case error: 8.88e-16   (precisión de punto flotante)
auc mine: 0.7327272727272728  auc vendor: 0.7327272727272728  diff: 0.0
```

Coincidencia exacta. El número que sigue en este informe es el mismo que
produciría `scoring_registration.py`.

### 1.2 Parametrización de la pose (4 DOF) y "mediana de una pose"

La GT de Task 2 es una similitud reflejada de 4 DOF (ver
`fido-task2-geometry` en memoria — matriz `[[s·cosθ, s·sinθ, tx], [s·sinθ,
-s·cosθ, ty], [0,0,1]]`, reflexión fija verificada en el 100% de los 1214
casos de train). `decompose_similarity` (ya en `src/fido/geometry.py`)
recupera `(tx, ty, cosθ, sinθ, s)` de cualquier matriz 3×3.

**No se promedia la matriz elemento a elemento.** Se descompone cada caso en
sus 4 parámetros y se agrega cada uno por separado:

- `tx`, `ty`, `s`: mediana/media/moda estándar sobre el conjunto de casos.
- `θ` (circular): θ = atan2(sinθ, cosθ). Se calcula la dirección media
  circular `θ̄ = atan2(mean(sinθ), mean(cosθ))`, se "desenvuelve" cada ángulo
  a su representante lineal más cercano a `θ̄` (dentro de `(θ̄-π, θ̄+π]`), y
  se toma la mediana/media/moda lineal de esos representantes, volviendo a
  envolver a `(-π, π]` al final. Es la mediana/moda circular estándar,
  válida mientras la muestra no cubra más de media vuelta — cierto aquí (ver
  §2, sd de θ por escenario entre 26° y 63°, muy por debajo de 90°).
- La "moda" de un parámetro continuo se estima con un KDE gaussiano 1D
  (`scipy.stats.gaussian_kde`) y se toma el argmáx sobre una grilla fina; para
  θ se aplica sobre el mismo desenvolvimiento que la mediana circular.
  **Caveat:** en algunas particiones (p.ej. Scenario_02, Scenario_08) el KDE
  de θ encontró un pico distinto y bastante alejado de la mediana/media
  (diferencias de 20–40°), señal de que la moda circular es menos estable
  que mediana/media sobre esta muestra. No cambia el veredicto: incluso el
  mejor caso con moda (oráculo por escenario) da AUC = 0.0000.

Con los 5 parámetros agregados se reconstruye la matriz vía
`compose_similarity(..., reflect=True)`.

### 1.3 Datos

- **Train**: `data/_annotations/Task 2`, 10 escenarios (`Scenario_01`…`10`),
  1214 casos totales (152, 154, 83, 127, 108, 121, 76, 145, 183, 65 por
  escenario). GT en `Ground Truth.Task 2` de cada JSON.
- **Mock Test**: `data/Mock Test/Task 2`, **1 solo escenario**
  (`Scenario_12`, no está entre los 10 de train), **5 casos**. Simula la
  situación real de submission: un escenario que el constante nunca vio.

Script reproducible:
`analysis/measure_task2_constant_pose_baseline.py` → escribe
`experiments/95-t2-constant-pose/results.json` (todos los números de este
informe, con parámetros exactos de cada constante evaluada).

```
python analysis/measure_task2_constant_pose_baseline.py
```

---

## 2. Por qué falla: solo la escala es estable, la posición y la rotación no

Antes de ver los resultados de AUC, esto explica el porqué. Dispersión
(media ± sd) de cada parámetro **dentro de cada escenario de train**:

| Escenario | n | tx (px) | ty (px) | escala (px) | θ (°) |
|---|---:|---|---|---|---|
| Scenario_01 | 152 | 551 ± 119 | 443 ± 157 | 172 ± 3.0 | 131 ± 44 |
| Scenario_02 | 154 | 494 ± 98  | 491 ± 121 | 174 ± 2.1 | 130 ± 44 |
| Scenario_03 | 83  | 528 ± 136 | 530 ± 159 | 157 ± 3.9 | 119 ± 63 |
| Scenario_04 | 127 | 533 ± 125 | 471 ± 137 | 169 ± 3.3 | 131 ± 38 |
| Scenario_05 | 108 | 468 ± 88  | 487 ± 113 | 129 ± 2.2 | 136 ± 26 |
| Scenario_06 | 121 | 473 ± 126 | 466 ± 130 | 151 ± 3.2 | 129 ± 39 |
| Scenario_07 | 76  | 449 ± 98  | 468 ± 129 | 159 ± 3.0 | 137 ± 27 |
| Scenario_08 | 145 | 487 ± 94  | 465 ± 126 | 159 ± 2.1 | 129 ± 45 |
| Scenario_09 | 183 | 367 ± 78  | 444 ± 138 | 151 ± 3.5 | 133 ± 27 |
| Scenario_10 | 65  | 334 ± 116 | 412 ± 172 | 178 ± 5.9 | 130 ± 47 |

La escala sí es casi constante dentro de cada escenario (sd 2.1–5.9 px sobre
una media ~130–178 px, el "5%" del indicio original). **Pero `tx` y `ty`
tienen sd de 78–172 px, y θ tiene sd de 26–63°, dentro del MISMO escenario.**
El Mock Test (5 casos, 1 escenario) lo confirma a escala pequeña: `tx` va de
77 a 361 px, `ty` de -96 a 300 px, θ de 130° a 167°, mientras la escala se
mueve solo entre 207 y 224 px (consistente con el rango [213.81, 226.16] del
indicio original — pero ese indicio era **solo sobre la escala**).

El error de esquinas que puntúa el challenge es dominado por la traslación:
un error de 100 px en `tx` mueve las 4 esquinas proyectadas ~100 px, muy por
encima del umbral máximo de 10 px de la métrica. Con `tx`/`ty` variando
80–170 px de sd incluso dentro de un mismo escenario, ninguna constante —ni
siquiera la oráculo, calibrada con el propio escenario de evaluación— puede
acercarse al umbral de 10 px en más que un puñado de casos.

---

## 3. Resultados — AUC oficial (0-1, umbrales enteros 0-10 px)

Los tres métodos de agregación (mediana, media, moda) dan prácticamente el
mismo resultado catastrófico; se reportan los tres para completar lo pedido.

| Medición | Mediana | Media | Moda |
|---|---:|---:|---:|
| [1] Constante global (1214→1214, referencia, ya descartada) | 0.0000 | 0.0000 | 0.0000 |
| [2] Constante por escenario, **oráculo** (pooled, 1214 casos) | **0.0001** | 0.0000 | 0.0000 |
| [3] **Leave-one-scenario-out** (pooled, 1214 casos) — HONESTO | **0.0000** | 0.0000 | 0.0000 |
| [4] Mock Test (5 casos, escenario nunca visto), constante = mediana/media/moda global | **0.0000** | 0.0000 | 0.0000 |

Error medio de esquinas (px) para las mismas filas — da la magnitud del
fracaso, útil porque la AUC ya está en el piso para todos:

| Medición | Mediana | Media | Moda |
|---|---:|---:|---:|
| [1] Global in-sample | 179.8 | 179.5 | 204.4 |
| [2] Oráculo por escenario (pooled) | 168.3 | 167.8 | 180.4 |
| [3] LOSO (pooled) | 183.0 | 182.5 | 211.7 |
| [4] Mock Test | 467.5 | 466.4 | 551.5 |

Fracción de casos con error ≤ umbral (mediana, la más favorable de las
tres) — muestra que ni siquiera con el oráculo se roza el umbral que puntúa:

| Umbral | [1] Global in-sample | [2] Oráculo pooled | [3] LOSO pooled |
|---:|---:|---:|---:|
| ≤10 px (el único que cuenta en la AUC de forma no despreciable) | 0/1214 (0.0%) | 1/1214 (0.08%) | 0/1214 (0.0%) |
| ≤25 px | 0.16% | 0.16% | 0.16% |
| ≤50 px | 2.2% | 2.6% | 2.2% |
| ≤100 px | 17.4% | 21.2% | 16.9% |
| ≤200 px | 61.3% | 67.0% | 60.2% |

Con el oráculo (calibrado y evaluado sobre el MISMO escenario, el mejor caso
imaginable para una constante) solo **1 de 1214 casos** cae por debajo de
10 px. La constante honesta (LOSO) no logra ni uno.

### 3.1 Desglose LOSO por escenario (mediana), el número que importa

| Escenario dejado fuera | n | AUC | error medio (px) |
|---|---:|---:|---:|
| Scenario_01 | 152 | 0.0000 | 209.6 |
| Scenario_02 | 154 | 0.0000 | 158.0 |
| Scenario_03 | 83  | 0.0000 | 221.8 |
| Scenario_04 | 127 | 0.0000 | 187.6 |
| Scenario_05 | 108 | 0.0000 | 143.6 |
| Scenario_06 | 121 | 0.0000 | 171.3 |
| Scenario_07 | 76  | 0.0000 | 155.9 |
| Scenario_08 | 145 | 0.0000 | 154.7 |
| Scenario_09 | 183 | 0.0000 | 201.2 |
| Scenario_10 | 65  | 0.0000 | 252.6 |

Cero AUC en los 10 escenarios, sin excepción. No es un problema de un
escenario atípico arrastrando el promedio: **ningún** escenario de train, sea
cual sea el que se deja fuera, permite que una constante entrenada en los
otros nueve puntúe.

### 3.2 Mock Test — simulación de submission real

`Scenario_12` (5 casos, no está en los 10 de train). Predicción = mediana
global de los 1214 casos de train.

| case_id | error de esquina (px) |
|---|---:|
| Scenario_12_00031 | 419.1 |
| Scenario_12_00066 | 425.7 |
| Scenario_12_00092 | 562.4 |
| Scenario_12_00099 | 422.5 |
| Scenario_12_00177 | 508.0 |

AUC = 0.0000, error medio 467.5 px. Ni un solo caso se acerca al umbral de
10 px. Es coherente con nuestro 0.00 actual en el leaderboard: una submission
constante hoy no nos movería de ahí.

---

## 4. Interpretación

**La hipótesis muere.** El criterio pre-registrado era: ≥0.15 → hipótesis
fuerte; <0.05 → hipótesis muerta. El resultado, en las tres variantes de
agregación y en las tres formas de medir (oráculo, LOSO, Mock Test), está
entre **0.0000 y 0.0001** — no en la zona intermedia, sino pegado a cero.

El indicio original (rango de escala del Mock Test dentro de un 5%) era
real pero incompleto: **la escala es efectivamente estable** dentro de un
escenario, pero la métrica oficial no perdona ningún otro grado de libertad.
La traslación (`tx`, `ty`) tiene una dispersión de 80–170 px de sd incluso
DENTRO de un único escenario (confirmado también en los 5 casos del Mock
Test: `tx` se mueve 77→361 px), muy por encima del umbral de 10 px que exige
la AUC. La rotación añade 26–63° de sd por su cuenta. Ninguna constante,
ni siquiera calibrada con oráculo sobre el propio escenario de evaluación,
puede compensar esa dispersión.

Esto no descarta que el leaderboard tenga *algún* componente de prior de
pose (p.ej. un modelo que aprenda "el fundus centra la mácula, la escala
está en tal rango" como *parte* de su predicción) — pero descarta que una
constante pura, sin mirar la imagen, explique un AUC de 0.475, 0.376 o
siquiera 0.244. Esos scores exigen que el modelo condicione la traslación y
la rotación por caso, es decir, señal real extraída de la imagen (visual o
de algún otro canal disponible en la inferencia, p.ej. metadata del volumen
OCT), no solo un prior de escenario.

**Consecuencia práctica:** no hay una submission trivial de "matriz
constante" que nos saque del 0.00 hoy. El camino sigue siendo encontrar
señal real (o replantear qué información aparte de vasculatura podría
correlacionar con la pose — el oráculo de apariencia ya se descartó por
separado, ver memoria `fido-task2-appearance-oracle-negative`).

---

## 5. Artefactos

- Script: `analysis/measure_task2_constant_pose_baseline.py` (reproducible,
  sin GPU, ~10 s).
- Resultados completos (todas las matrices/params por método y por
  escenario): `experiments/95-t2-constant-pose/results.json`.
- Este informe: `experiments/95-t2-constant-pose/INFORME.md`.
