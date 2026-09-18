# T2-89 — ¿El instrumento sirve como landmark compartido para registrar?

> **Actualización:** el gate pre-registrado en la sección 5 ya se corrió.
> Resultado: **NO-GO** (AUC extrapolado a 1214 casos = 0.0004, error medio
> 935 px). Ver `GATE_RESULTS.md` en este mismo directorio para el comando
> exacto, la salida real y el diagnóstico de por qué falla.

**Encargo:** evaluar en serio si la señal de instrumento del oráculo T2-R11/R11b
(`experiments/83-t2-enface-convention/`) es explotable como landmark cross-modal
para resolver la similitud de 4 DOF de Task 2, más allá de su uso original como
control positivo del arnés.

**Estado:** análisis solo-CPU, sin GPU, sin entrenar nada. Todo número de esta
página o se cita de un archivo existente (archivo:línea) o se recalculó aquí
con un script nuevo cuyo comando y log quedan en este mismo directorio.

**Veredicto corto:** la señal es real, no es un artefacto de la convención de
ejes — pero es más delgada y más frágil de lo que sugiere la tabla del
oráculo. Cobertura real: **22.98 % de los casos** (no ~30 %), repartida de
forma **binaria por escenario** (0 % en 6/10 escenarios, ~50 % en 4/10), y el
único chequeo fuera de distribución disponible (Mock Test, n=5) **no
reproduce** el enriquecimiento a los radios que el oráculo usó para declarar
el gate. Van dos piezas de infraestructura que no existen todavía (segmentador
de instrumento en el volumen OCT; segundo keypoint fundus). Recomendación:
**GO condicional** a un gate offline de una tarde (sección 5) antes de
comprometer una sola hora de GPU.

---

## 1. Qué mide exactamente el oráculo, y si hay artefacto

### 1.1 Mecanismo exacto

`analysis/verify_task2_enface_convention.py` hace, por caso:

1. `points_from_group()` (línea 43) lee del JSON de anotación las coordenadas
   **fundus** (`microscope.png`, 1024×1024) de hasta 6 puntos de instrumento:
   `Endgripping Forceps` → `Right Head Tip`, `Left Head Tip`, `Joint Tip`,
   `Start`; `Endoilluminator` → `Tip`, `Start`.
2. `map_fundus_to_uv()` (línea 37) proyecta esos puntos a coordenadas
   normalizadas `uv ∈ [0,1]²` invirtiendo la **matriz GT de Task 2** de ese
   mismo caso (`data["Ground Truth"]["Task 2"]`).
3. `inside_uv()` (línea 50) descarta los puntos que caen fuera del cuadrado
   unitario. Los que sobreviven son el `n` que aparece en la tabla del
   oráculo: **312** puntos en el fold de train (966 casos,
   `experiments/83-t2-enface-convention/oracle_train.json:instrument_points`),
   **469** en el conjunto completo de 1214 (`ATTACK_LADDER.md:518`, y
   reproducido de forma independiente abajo).
4. `transform_uv()` (línea 57) aplica una de las 8 simetrías de eje
   (transpose × flip_u × flip_v) — la ganadora congelada es
   `transpose__flip_u__flip_v` (`src/fido/data/common.py:112`).
5. `measure_points()`/`unit_square_distance()` (líneas 74-90) calculan, para
   cada punto transformado, la distancia (en fracción del lado del cuadrado
   unitario) al píxel **segmentado como instrumento** más cercano en la
   proyección en-face real del volumen OCT (`seg` con clases
   `INSTRUMENT_CLASSES = (8, 10, 11)`, línea 24 — máscaras GT, no predichas).
6. Se compara esa tasa de "distancia ≤ r" contra un null de 200 puntos
   uniformes por punto real, transformados por el mismo camino
   (`aggregate()`, línea 124). El ratio observado/null es un **enriquecimiento
   espacial pooled**, no un p-valor.

### 1.2 ¿Es sólida o hay artefacto?

**No es un artefacto de convención ni de etiquetado.** La prueba de esto no es
el ratio en sí, sino la **selectividad**: de las 8 simetrías posibles, solo
`transpose__flip_u__flip_v` da un enriquecimiento sostenido en los 4 radios
(3.44 / 3.40 / 3.60 / 3.01 en train-fold,
`experiments/83-t2-enface-convention/oracle_train.md`); el resto oscila entre
0.23 y 2.03. Y la vasculatura, medida con el **mismo código, el mismo caso,
la misma matriz**, se queda plana (0.81–1.10) bajo las 8 variantes por igual.
Si el hallazgo fuera ruido o un bug geométrico, no se vería esta asimetría
entre instrumento y vasos, ni la selección de una sola variante entre ocho.

**Pero hay dos matices que la lectura original no hace explícitos, y que
importan para decidir si esto es "landmark usable":**

- **(a) Usa la matriz GT que se quiere recuperar.** El paso 2
  (`map_fundus_to_uv`) proyecta el punto de fundus a `uv` invirtiendo la
  matriz verdadera de ese caso. Esto es legítimo para responder *"¿existe
  correspondencia física entre dónde está el instrumento en el fundus y dónde
  está en el volumen OCT, dado el registro correcto?"* — y la respuesta es sí.
  Pero **no** responde *"¿puedo recuperar esa correspondencia detectando el
  instrumento de forma independiente en cada modalidad?"*, que es lo que un
  solver necesita en inferencia. Son preguntas distintas; el oráculo solo
  contesta la primera.
- **(b) La muestra es más delgada de lo que aparenta.** 312–469 puntos suena
  a bastante, pero salen de **como mucho 3 puntos por caso** (ver §2) y de una
  **minoría de casos** que además están correlacionados dentro de escenario
  (frames consecutivos de la misma maniobra quirúrgica). El enriquecimiento
  "sostenido en los 4 radios" es real, pero el tamaño de muestra efectivo es
  bastante menor que el `n` crudo sugiere.

**El chequeo fuera de muestra que sí hicimos (Mock Test, n=5, único
escenario nuevo disponible) no reproduce el enriquecimiento** — ver §4.3. Con
n=5 esto no refuta el hallazgo de train, pero es una señal de alerta real que
el proyecto no puede permitirse ignorar dado el historial de leer tablas a
medias.

---

## 2. Cobertura real (recalculada, no asumida)

Comando y log completos en
`experiments/89-t2-instrument-landmark/coverage_train_annotations.log`, script
`analysis/measure_task2_instrument_coverage.py`, corrido contra las 1214
anotaciones locales (`data/_annotations/Task 2/`, mismos JSON que usa
`find_task2_cases`, sin necesitar el volumen OCT porque este paso solo lee
`Ground Truth/Task 2` + `Keypoints`).

```
casos totales (json con GT Task 2): 1214
casos con >=1 keypoint de instrumento DENTRO de la huella uv: 279/1214 (0.2298)
suma de puntos dentro de la huella: 469
distribución de puntos-dentro-de-huella por caso:
  0 puntos: 935 casos
  1 puntos: 152 casos
  2 puntos: 64 casos
  3 puntos: 63 casos
casos con >=2 puntos (Procrustes/Umeyama cerrado, 4 DOF): 127/1214 (0.1046)
casos con ==1 punto (solo traslación): 152/1214 (0.1252)
```

Esto reproduce exactamente `279/1214 (0.2298)` que ya estaba documentado en
`ATTACK_LADDER.md:518` (T2-R11) — cross-check limpio de que la metodología es
la misma.

**El número que faltaba y que cambia la lectura:** el techo de un solver que
resuelva los 4 DOF de golpe (posición + rotación + escala, vía Procrustes con
≥2 correspondencias) NO es 23 % — es **10.46 % (127/1214)**. El 12.52 %
restante (152/1214, exactamente 1 punto) solo da traslación (2 DOF); rotación
y escala tendrían que venir de un prior externo, y el prior externo de escala
ya se probó y **no transfiere fuera de distribución** (T2-81/T2-86, ver §3).

**Cobertura no es uniforme por caso — es casi binaria por escenario:**

| Escenario | Casos con instrumento en huella | Escala media (px) |
|---|---:|---:|
| 01 | 84/152 = 55.3 % | 172.19 |
| 02 | 3/154 = 1.9 % | 173.89 |
| 03 | 0/83 = 0.0 % | 157.22 |
| 04 | 61/127 = 48.0 % | 169.84 |
| 05 | 0/108 = 0.0 % | 129.00 |
| 06 | 0/121 = 0.0 % | 150.95 |
| 07 | 37/76 = 48.7 % | 159.25 |
| 08 | 0/145 = 0.0 % | 159.21 |
| 09 | 94/183 = 51.4 % | 150.42 |
| 10 | 0/65 = 0.0 % | 177.14 |

Es decir: en 6 de los 10 escenarios de train el instrumento **prácticamente
nunca** entra en la huella en-face del OCT; en los otros 4 entra en ~la mitad
de los frames. Esto no es "una señal débil pero uniforme que ayuda un poco en
todos lados" — es "una señal fuerte que solo existe en ciertos escenarios
quirúrgicos". El test oculto tiene una composición de escenarios desconocida:
**no se puede saber, sin correrlo, si la vía de instrumento cubrirá 0 % o 50 %
del test real.**

---

## 3. DOF que resuelve, y el problema de la escala

- **1 punto → 2 DOF** (`tx, ty`), asumiendo que rotación/escala vienen de otra
  parte. 152/1214 casos (12.52 %).
- **≥2 puntos → 4 DOF completos** vía Procrustes/Umeyama en forma cerrada
  (traslación + rotación + escala en un solo ajuste de mínimos cuadrados, sin
  pasos separados de "eje → rotación" y "otra cosa → escala"). 127/1214 casos
  (10.46 %). Esto corrige el encuadre del encargo: no hace falta cerrar la
  escala "de otra forma" cuando hay ≥2 correspondencias — el propio Procrustes
  la resuelve al mismo tiempo que la rotación, con la ventaja añadida de que
  la estima **por caso**, no desde un prior poblacional.

**Por qué esto importa dado lo que ya se sabe de la escala** (verificado de
forma independiente, coincide con `NOW.md:213-217` a los dos decimales):

```
GLOBAL (n=1214): mean=159.80  sd=13.59  min=126.66  max=188.09
Rango de medias por escenario: [129.00, 177.14]
sd dentro de escenario: 1.82 - 4.15 px
Mock Test (Scenario_12, n=5, medido localmente): min=213.81 max=226.16
```

- La escala es casi constante **dentro** de un escenario pero varía **entre**
  escenarios — es una variable de nivel-escenario, no ruido de sensor.
- `properties.json` es idéntico en los 10 escenarios de train y en el Mock
  (`THE_MAP.md:216`, `NOW.md:216`) — **no hay metadato que dé la escala**, ya
  se descartó esa vía.
- El Mock Test (única muestra fuera de distribución) tiene escala
  **completamente fuera** del rango de train — confirmado de nuevo aquí sobre
  los 5 casos locales, no solo citado. Un prior de escala aprendido en train
  (media/rango poblacional) **no cubre** el régimen del test; esto ya está
  diagnosticado como causa probable del colapso a 0.00
  (`experiments/86-t2-postmortem/INFORME.md:85-86`, T2-81 no transfirió).

**Conclusión de esta sección:** un solver de ≥2 puntos es la única vía
identificada hasta ahora que estima la escala **por caso, desde la propia
evidencia del caso**, sin depender de un prior poblacional que ya se sabe que
no generaliza. Eso es genuinamente valioso — pero solo aplica al 10.46 % de
los casos, y ese 10.46 % está concentrado en escenarios específicos (§2).

---

## 4. ¿Se puede construir un solver directo? Piezas, huecos, presupuesto de error

### 4.1 Piezas que SÍ existen

- **Punta de instrumento en fundus**: `src/fido/models/task1_keypoint_resnet.py`
  entrenado sobre `Ground Truth/Task 1` = `[x, y, distancia]`, con `x,y` en el
  mismo espacio de píxeles de `microscope.png` (1024×1024) que usa la matriz
  de Task 2 (`src/fido/data/task1.py:284-286,329`;
  `RULES_OF_ENGAGEMENT.md:65`). Resultado medido:
  `val_keypoint_auc=0.8539`, `val_mean_error≈1.03 px` en época 5
  (`ATTACK_LADDER.md:915`).
- **Máscaras GT de instrumento en fundus**: `Stereo Left/<frame>/Segmentation/
  forceps.png` y `endoilluminator.png` (`THE_MAP.md:151-155`) — no hay modelo
  entrenado sobre ellas todavía, pero el material de entrenamiento existe si
  se quisiera un segundo punto (o el eje) del lado fundus.
- **Máscaras GT de instrumento en el volumen OCT**: clases 8/10/11 en
  `Volume/<frame>/Segmentation/*.png` — es lo que usó el oráculo, pero es
  anotación, **no** algo disponible en inferencia (`oct_volume` en inferencia
  es el array crudo, `RULES_OF_ENGAGEMENT.md:69-70`).

### 4.2 Piezas que NO existen (trabajo nuevo, no trivial)

1. **Segmentador/detector de instrumento en el volumen OCT en-face.** Es la
   pieza que falta para convertir "sabemos dónde debería estar" en "podemos
   encontrarlo sin la anotación". No hay nada de esto en el repo hoy
   (`THE_MAP.md`, listado de modelos, no menciona segmentación de instrumento
   OCT como modelo entrenado).
2. **Segundo punto en fundus** (para pasar de 1 a 2 correspondencias en los
   casos que solo tienen 1 dentro de la huella, y para tener redundancia en
   los que ya tienen ≥2). Task 1 da un único keypoint por diseño.

### 4.3 Chequeo de sanidad fuera de muestra (Mock Test, n=5) — resultado que templa el optimismo

Comando y log completos:
`experiments/89-t2-instrument-landmark/mock_sanity_check.log`
(`analysis/check_task2_instrument_mock_sanity.py`, reusa `measure_case` tal
cual, sin modificar nada del código del oráculo).

```
Scenario_12/00031: c_count=2  distancias=[0.158, 0.270]
Scenario_12/00066: c_count=2  distancias=[0.066, 0.121]
Scenario_12/00092: c_count=3  distancias=[0.727, 0.919, 0.979]
Scenario_12/00099: c_count=1  distancias=[0.046]
Scenario_12/00177: c_count=0
```

Los radios que el oráculo usó para declarar el gate en train van de 0.005 a
0.04. **Ninguna de las 8 distancias observadas en Mock cae dentro de 0.04** —
la más cercana es 0.046, justo por fuera. En unidades de píxel-fundus
(distancia × escala ≈ 214-226 px), esto es del orden de **10 a 220 px**, muy
por encima de la ventana de 0-10 px que puntúa la métrica de Task 2.

**Con n=5 esto no refuta el hallazgo de train** (demasiado poca potencia), pero
tampoco lo confirma fuera de distribución, que es justo la pregunta que
importa. Es la advertencia más concreta de todo este informe: el instrumento
SÍ entra en la huella del OCT en 4/5 casos de Mock (coherente con estar en un
escenario de "alta cobertura" como 01/04/07/09), pero cuando entra, no
aterriza cerca de donde está segmentado el instrumento a la escala de
precisión que la métrica exige.

### 4.4 Presupuesto de error (orden de magnitud, no medición)

Con ≥2 correspondencias, Procrustes/Umeyama da el ajuste de mínimos cuadrados
exacto — el error final lo determina el ruido de localización de los puntos,
no el solver:

- Techo de referencia: incluso con la matriz GT descompuesta y reensamblada
  perfectamente en el modelo de 4 DOF (traslación+rotación+escala exactos),
  el residuo medio es **3.48 px**, AUC **0.6389** (`ATTACK_LADDER.md:567`,
  T2-80 "TODO GT"). Ese es el techo absoluto de *cualquier* solver de 4 DOF,
  instrumento o no — muy por encima del 0.00 actual y del líder del
  leaderboard (0.475).
- Un error de punto de `Δuv` (fracción del cuadrado unitario) se traduce en
  aproximadamente `Δuv × escala_px` de error en espacio fundus (escala ≈
  130-225 px según caso). Un "acierto cercano" según el propio oráculo
  (`Δuv≈0.02`) ya cuesta **~3.2 px** — comparable a TODO el residuo del techo
  T2-80 con **cero** ruido adicional de detección.
- Lo que realmente se observó en el único chequeo OOD disponible
  (`Δuv` entre 0.05 y 0.98) se traduce en **10-220 px** — muy por fuera de la
  ventana 0-10 px de la métrica. Si esa dispersión es representativa (n=5, no
  se puede afirmar que lo sea), un solver de instrumento en esos casos
  **empeoraría** el baseline en vez de ayudarlo.
- Localización fundus (Task 1, ~1 px) es la parte barata y ya resuelta.
  Localización OCT (segmentador que no existe) es la incógnita que domina el
  presupuesto de error, y el único dato que tenemos sobre su dificultad
  (§4.3) no es alentador.

---

## 5. Veredicto y peldaño

**No es una vía de solución completa de Task 2**: el techo de "resolver los 4
DOF solo con instrumento" es 10.46 % de los casos (127/1214), concentrado en
4 de 10 escenarios, con una composición del test oculto desconocida. No
reemplaza vasculatura ni apariencia como vía principal.

**Pero el proyecto está en 0.00.** Cualquier vía que toque de forma
verificable un subconjunto no trivial de casos, sin empeorar el resto, es un
peldaño legítimo — no necesita resolver el 100 % de los casos, necesita (a)
no dañar los casos donde no aplica y (b) genuinamente ayudar donde sí aplica.
Dado lo medido en §4.3, ese "sí ayuda" todavía no está demostrado.

### Hipótesis única (T2-89)

Para el subconjunto de train con ≥2 keypoints de instrumento dentro de la
huella en-face (127/1214 casos, lista exacta reproducible con
`analysis/measure_task2_instrument_coverage.py`), resolver la similitud de 4
DOF por Procrustes/Umeyama usando **correspondencias GT-a-GT** (keypoints
anotados de fundus + centroide/eje-mayor-PCA de la máscara GT de instrumento
en OCT) debe acercarse al techo TODO-GT de T2-80 (AUC 0.6389) **sobre ese
subconjunto**, muy por encima del baseline actual.

Esto es deliberadamente un oráculo GT-a-GT, no un modelo — el objetivo es
aislar "¿la geometría alcanza la precisión que pide la métrica?" de "¿podemos
construir los detectores?", antes de gastar una sola hora de GPU en el punto
4.2.

### Protocolo y gate (CPU, sin entrenar, mismo día)

1. Restringir a los 127 casos con ≥2 puntos de instrumento dentro de la
   huella (train, todos los escenarios que aportan cobertura: 01/04/07/09 +
   los pocos casos de 02).
2. Por caso: tomar los puntos GT de fundus ya usados por el oráculo
   (`points_from_group` + `inside_uv`); del lado OCT, derivar puntos
   equivalentes de la máscara GT de instrumento (p. ej. centroide + extremo
   del eje mayor vía PCA, o los propios puntos si hay anotación 3D
   proyectable — a decidir en la implementación).
3. Resolver Procrustes/Umeyama (similitud con reflexión fija, coherente con
   `det<0` ya confirmado) y reconstruir `M_predicha`.
4. Puntuar con la misma definición de corner-AUC que usa el scoring oficial
   (`RULES_OF_ENGAGEMENT.md:119-121`), restringido a esos 127 casos.

**GO** si AUC-subconjunto ≥ 0.30 (la mitad del techo T2-80, dejando margen
2× para el ruido que meterán los detectores reales) **y** error medio de
esquina ≤ 15 px.

**NO-GO** si AUC-subconjunto < 0.10 o error medio > 40 px — significaría que
ni siquiera con anotaciones perfectas de ambos lados el punto derivado de
máscara (centroide/PCA) alcanza la precisión que exige la métrica; en ese
caso el bloqueo no es "faltan detectores", es que la geometría del landmark
no alcanza, y se cierra la vía sin gastar GPU.

**Repetir el mismo oráculo GT-a-GT sobre los 3 casos de Mock con ≥2 puntos**
(00031, 00066, 00092) como chequeo OOD barato antes de decidir — dado lo
visto en §4.3, es razonable esperar que Mock salga peor que train; el gate
GO/NO-GO se decide sobre train, pero el número de Mock debe registrarse igual
por honestidad experimental (Constitución, regla 5).

**Costo:** un script CPU, un par de horas, cero GPU, cero entrenamiento. Solo
si este gate pasa se autoriza la siguiente fase (entrenar segmentador de
instrumento en OCT + segundo keypoint fundus), que sí es trabajo de GPU nuevo
y no debe empezar antes.

---

## Archivos de este rung

- `analysis/measure_task2_instrument_coverage.py` — cobertura por caso y por
  escenario sobre las 1214 anotaciones de train (nuevo).
- `analysis/check_task2_instrument_mock_sanity.py` — sanity check GT-a-GT
  sobre los 5 casos de Mock Test, reusa `measure_case` sin modificarlo
  (nuevo).
- `experiments/89-t2-instrument-landmark/coverage_train_annotations.log` —
  salida real del primer script.
- `experiments/89-t2-instrument-landmark/mock_sanity_check.log` — salida real
  del segundo script.

## Lo que NO se verificó (para que quede explícito)

- No se corrió nada sobre GPU ni se entrenó ningún modelo.
- No se midió el error de un segmentador de instrumento OCT real (no existe);
  el presupuesto de error de §4.4 es una estimación de orden de magnitud, no
  una medición.
- No se confirmó que el keypoint GT de Task 1 sea el mismo punto físico que
  los grupos `Endgripping Forceps`/`Endoilluminator` de Task 2 — son conjuntos
  de snapshots distintos (zips separados); la reutilización del modelo de
  Task 1 es plausible, no validada.
- El chequeo de Mock Test (§4.3) tiene n=5 (8 puntos). No alcanza para
  refutar ni confirmar nada por sí solo; se reporta porque es el único dato
  fuera de distribución disponible y contradice la lectura optimista.
