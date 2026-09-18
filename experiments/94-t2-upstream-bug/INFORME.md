# T2-94 — Auditoría del pipeline en-face: ¿dónde está nuestro error?

**Encargo:** el leaderboard público muestra 7 equipos resolviendo Task 2
(cnielsen 0.475, techo GT 0.6389) mientras nosotros cerramos 5 vías internas
por "ausencia de señal". Esa contradicción implica un error en nuestra propia
cadena de medición. Encontrarlo.

**Veredicto corto:** **no encontramos un bug de código, convención o
geometría.** Todo lo auditado aquí — construcción de la proyección en-face,
convención de ejes, matriz GT contra el scorer oficial, y auto-consistencia
del oráculo — pasa limpio, con evidencia nueva y no solo repitiendo lo
existente. La contradicción documental que disparó este encargo (NOW.md
15-ago "se ve vasculatura alineada" vs. oráculo "vasculatura no correlaciona")
**se resuelve, no se cierra en falso**: lo que se ve a simple vista es real
como percepción pero no es vasculatura de precisión-píxel — es una ilusión
gestalt reforzada por sombras de instrumento y artefactos de baja frecuencia.
La vía de instrumento, que sí tiene una señal real y medible, ya fue auditada
con rigor por T2-89 (oráculo GT-a-GT, no un modelo) y cerrada por un margen de
3 órdenes de magnitud — esa auditoría se sostiene bajo un segundo examen aquí.
El misterio de cómo el líder llega a 0.475 **queda abierto**, no resuelto por
esta auditoría — ver §6.

Todo lo de abajo corre 100% local, CPU, sin pod: 5 casos reales de train
extraídos selectivamente de `data/Task 2/Scenario_0{1,3}.zip` (sin
descomprimir los ~3.6 GB completos) más sus anotaciones ya locales en
`data/_annotations/Task 2/`. Scripts nuevos en este directorio; nada en
`vendor/fido/` ni en `src/fido/` se tocó.

---

## 1. Verificación visual de la proyección en-face

**Script**: `experiments/94-t2-upstream-bug/render_inspection.py`.
**Renders**: `experiments/94-t2-upstream-bug/renders/Scenario_0{1,3}_*.png`
(5 casos, grid completo por caso: fundus + huella, mean/MIP/min, 8 slabs de
profundidad, y máscaras GT de vaso/instrumento).

Casos inspeccionados: `Scenario_01/{00000,00030,00075}`,
`Scenario_03/{00000,00040}` — reales, no Mock Test, extraídos vía `zipfile`
selectivo (`iOCT Microscope/Volume/<frame>/` + `Stereo Left/<frame>/`).

### 1.1 Lo que se ve

`Scenario_03/00040` es el caso más claro (`FIG_A_vessel_comparison.png`):

- La proyección **mean full-depth** (la que usa el pipeline hoy,
  `_enface_from_volume` / `enface_projection`) muestra curvas oscuras
  serpenteantes que **sí recuerdan a vasos**, más dos manchas oscuras en
  paralelogramo — esas dos manchas son la **sombra del instrumento** (el caso
  tiene un fórceps visible en el fundus), no vasculatura. Esto confirma por
  qué a simple vista (NOW.md, 15-ago) alguien concluyó "se ve vasculatura
  alineada": hay estructura real y visualmente convincente en esa imagen —
  solo que una parte importante de ella es sombra de instrumento, no vaso.
- La **máscara GT de clase vaso** (`seg==3`, ANY sobre los 512 px de
  profundidad) sí dibuja un árbol de ramificación con forma de vaso real
  (ver panel derecho de `FIG_A_vessel_comparison.png`) — no es ruido.
- Comparada directamente contra la máscara GT de vasos del **fundus**
  (`arteriesorveins.png`) proyectada al mismo grid con la matriz GT y la
  convención congelada (`transpose__flip_u__flip_v`), la topología gruesa
  *sugiere* similitud (un tronco principal con 2 ramas descendentes en ambas)
  pero el solape píxel a píxel es débil: **IoU con solape cero-shift entre
  0.015 y 0.038** en los 4 casos con segmentación de volumen disponible, y
  una búsqueda exhaustiva de desplazamiento (`dy∈[-30,30]`, `dx∈[-60,60]`
  píxeles nativos) solo sube el mejor IoU a **0.036–0.070** — sin un
  desplazamiento consistente entre casos (mejor `dy` fue -16, -30, +12 en
  tres casos). Esto es indistinguible de dos patrones de líneas de densidad
  similar sin relación real.

**Conclusión de esta sección**: la contradicción NOW.md-vs-oráculo es real
pero no es un bug — es la diferencia entre "se percibe similar" (cierto, y
explicable) y "correlaciona con precisión de píxel" (falso, confirmado de
nuevo aquí con máscaras GT, no solo con intensidad cruda).

### 1.2 Instrumento: la señal SÍ es visible directamente

`Scenario_01/00030` (3/6 keypoints de instrumento caen dentro de la huella
en-face): `renders/_overlay_instrument_S01_00030.png` superpone en **verde**
la máscara GT de instrumento del volumen OCT (clases 8/10/11, ANY sobre
profundidad) y en **rojo** la máscara GT de fórceps+endoiluminador del fundus
(`forceps.png`+`endoilluminator.png`), warpeada con la matriz GT y la misma
convención. Los dos blobs verdes quedan **pegados al borde** de la banda roja,
uno con solape directo (amarillo). Esto es consistente con el hallazgo ya
documentado del oráculo (`experiments/83-t2-enface-convention/oracle_train.md`:
instrumento, ratio observado/null 3.0–3.6× en las 4 radios, `gate_pass=true`)
y con el diagnóstico ya existente de T2-89 (§5): la señal es real pero
**dispersa e inconsistente entre casos** — este caso concreto es uno de los
"buenos" (T2-89 midió en Mock distancias tan buenas como 0.046-0.066 unidades
de lado, junto a otras tan malas como 0.98).

---

## 2. Auditoría de la proyección en-face: ¿qué eje, qué reducción?

**Confirmado por código** (`src/fido/data/common.py:102-109`,
`submissions/r06-fallback-fixed/inference.py:364-372`,
`infra/precompute_task2_enface.py:49`): las tres rutas de producción llaman a
la misma fórmula, `volume.mean(axis=1)`, sobre un volumen `(128, profundidad,
ancho)` — se promedia sobre el eje de **profundidad completa** (eje 1, tamaño
512 en los casos medidos aquí), no sobre slices ni sobre una banda. Ningún
desacuerdo entre las tres implementaciones.

### 2.1 ¿Diluye el promedio completo la señal de vaso? — Sí y no

Se construyó un barrido de 16 bandas de profundidad (32 px cada una, sobre
512) usando la **máscara GT de clase vaso** (no intensidad cruda) en los 4
casos con segmentación de volumen, midiendo densidad de cada banda y su IoU
contra la máscara de vaso del fundus warpeada:

```
Scenario_03/00040  fundus_vessel_frac=0.0622
  bandas 0-5   [0:192]    density=0.0000           <- vítreo/ILM, sin vaso
  bandas 6-11  [192:384]  density 0.0026-0.0166     <- capa donde vive el vaso
    mejor banda: banda 10 [320:352] IoU=0.0130
  bandas 12-15 [384:512]  density=0.0000-0.0016     <- coroides/esclera, sin vaso

Scenario_01/00000  fundus_vessel_frac=0.0822
    mejor banda: banda 10 [320:352] IoU=0.0220

Scenario_01/00030  fundus_vessel_frac=0.0434
    mejor banda: banda 9  [288:320] IoU=0.0071 (todas <0.006 salvo esta)

Scenario_03/00000  fundus_vessel_frac=0.0741
    mejor banda: banda 7  [224:256] IoU=0.0155
```

**Sí hay una banda de profundidad anatómicamente correcta**: la clase vaso
está estrictamente confinada a un rango medio de la profundidad (~índice
190–410 de 512; cero en vítreo y cero en coroides/esclera), lo cual tiene
sentido — es la capa de retina. Reducir por `ANY`/`mean` sobre las bandas
vacías no puede introducir señal falsa; en ese sentido el promedio completo
no está "contaminado" por profundidades irrelevantes de forma azarosa.

**Pero no hay una banda oculta con señal fuerte**: incluso la mejor banda
individual en cada caso da IoU 0.007–0.022, **del mismo orden o peor** que
`ANY` sobre la profundidad completa (0.015–0.038, sección 1.1). Es decir: la
hipótesis "promediar todo diluye una señal de vaso que en una banda angosta
sería fuerte" **no se sostiene** cuando se mide contra la máscara GT de vaso
real — no existe esa banda fuerte que descubrir. Esto refina, no contradice,
la sospecha original del encargo: el eje y la reducción están bien pensados
geométricamente, pero el problema no es la reducción — es que la propia
etiqueta de vaso del volumen OCT, a cualquier profundidad, no coincide en
posición con la etiqueta de vaso del fundus con la precisión que exige la
métrica (confirma, con máscaras en vez de keypoints, el resultado ya medido
por el oráculo de T2-R11/R11b sobre 1214/966 casos).

**Matiz que sí importa para intensidad cruda** (no para la clase vaso):
`analysis/measure_task2_oracle_signal.py` (`projection_images`, ya existente)
mide NCC de intensidad para mean/MIP/min/4-slabs/gradiente contra el fundus
completo. Repetido aquí para los 5 casos nuevos, el NCC de **mean full-depth**
varía de -0.14 a **+0.50** según el caso, y un slab de profundidad concreto
(banda 4/8, aprox. mitad de la retina) llega a **+0.59** en el mejor caso —
pero esta correlación de intensidad-cruda no es lo mismo que correlación con
la clase vaso (sección 1.1 y 2 muestran que la clase vaso, aisladamente, no
correlaciona). La lectura más consistente con toda la evidencia: ese NCC
positivo de intensidad está dominado por confusores de baja frecuencia
(sombra de instrumento, viñeteado, brillo general) — exactamente lo que
`T2-82` (CNN de descriptores comunes, con ablation de OCT-shuffle) ya
demostró que un modelo entrenado **no llega a explotar** (el shuffle de OCT
entre casos no degrada el loss, `RESULTS.md:237`).

---

## 3. Auto-consistencia del oráculo: test sintético

**Script**: consola, ver comandos en `render_inspection.py` (funciones
reutilizadas) — no se creó un script separado porque el test es corto y no
aporta un artefacto reutilizable más allá de este informe.

**Diseño**: se toma el fundus real de `Scenario_03/00040`, se construye una
"en-face sintética" **por warp exacto del propio fundus** con la matriz GT
real y la convención congelada — por construcción, corresponde perfectamente
en la pose correcta. Se mide luego NCC contra sí misma en la pose correcta, en
20 poses con traslación aleatoria, y en un barrido de desplazamiento
horizontal creciente:

| Pose | NCC |
|---|---:|
| Correcta (`dx=0`) | **1.0000** |
| 20 traslaciones aleatorias (`U(0,1023)²`) | media **-0.166**, std 0.459, rango [-0.837, 0.567] |
| `dx=5px` | 0.9565 |
| `dx=10px` | 0.8694 |
| `dx=20px` | 0.6511 |
| `dx=40px` | 0.3937 |
| `dx=80px` | 0.1218 |

**Resultado**: el oráculo detecta correspondencia perfecta en la pose
correcta (NCC=1.0 exacto) y degrada de forma suave y monótona con el error de
pose, cayendo a ruido para poses aleatorias. **La maquinaria de medición
(matriz→uv, convención de ejes, `warp_fundus`, NCC) no está rota** — si
hubiera señal real de vasculatura del tamaño que sugiere la inspección
visual, este oráculo la detectaría.

Adicionalmente, el **self-test geométrico** ya existente en
`analysis/measure_task2_oracle_signal.py::crosshair_error` (compara las
esquinas que implica la matriz GT contra los keypoints reales
`iOCT Microscope Crosshair` que el simulador anota independientemente) se
repitió sobre los 5 casos nuevos:

```
Scenario_01/00000  crosshair_error = 2.28e-05 px
Scenario_01/00030  crosshair_error = 2.22e-05 px
Scenario_01/00075  crosshair_error = 3.74e-05 px
Scenario_03/00000  crosshair_error = 3.53e-05 px
Scenario_03/00040  crosshair_error = 1.74e-05 px
```

Error de precisión de punto flotante, no de convención. La convención de ejes
grabada en `TASK2_ENFACE_CONVENTION = "transpose__flip_u__flip_v"`
(`src/fido/data/common.py:112`) es correcta.

---

## 4. Matriz GT y sistema de coordenadas contra el scorer oficial

Comparado línea por línea `vendor/fido/Codabench Bundle/scoring_program/
scoring_registration.py` contra nuestro código:

- **Carga de la referencia** (`scoring_registration.py:69-77`,
  `load_reference`): lee `data["Ground Truth"]["Task 2"]` directo del JSON de
  anotación como matriz `(3,3)`, **sin ningún reescalado ni ajuste**. Idéntico
  a como la cargan `measure_task2_oracle_signal.py`,
  `verify_task2_enface_convention.py` y `render_inspection.py` (este informe).
- **Esquinas de referencia** (`scoring_registration.py:14-20`,
  `REF_CORNERS_H`): cuadrado unitario en orden `(0,0),(1,0),(1,1),(0,1)`,
  proyectado por `H` con `project_corners` (`scoring_registration.py:48-54`).
  Reimplementado aquí en `render_inspection.py::project_corners`, verificado
  numéricamente idéntico.
- **Verificación independiente contra un tercer dato**: las esquinas que
  implica `H` en `(u,v)=(0,1)` y `(u,v)=(1,0)` coinciden — a **~2e-5 px**, no
  por aproximación — con los puntos `"Start 0"` y `"End 1"` del grupo
  `Keypoints/iOCT Microscope Crosshair` del JSON, un dato **anotado de forma
  independiente** por el simulador, no derivado de `H`. Esta coincidencia de
  tres fuentes distintas (nuestra interpretación de `H`, el scorer oficial, y
  la anotación de crosshair) es la prueba más fuerte disponible de que el
  sistema de coordenadas está bien entendido.
- **Unidades**: corner error se computa y puntúa en **píxeles del fundus**
  (`microscope.png`, 1024×1024) directamente — no hay paso de normalización
  oculto ni conversión de unidades de profundidad (eso solo aplica a Task 1,
  ver `NOW.md`, corrección de resolución del 19-ago). Confirmado:
  `MAX_THRESHOLD_PX = 10` en `scoring_registration.py:14`, consistente con
  `RULES_OF_ENGAGEMENT.md`.

**No se encontró ninguna discrepancia** entre nuestra interpretación de la
matriz GT y la del scorer oficial.

---

## 5. Lo que ya se había auditado (y se sostiene bajo este segundo examen)

Este informe no reemplaza el trabajo anterior — lo pone a prueba de nuevo con
método distinto (visual + máscaras GT en vez de solo keypoints/intensidad) y
lo confirma:

- **T2-R11/R11b** (`experiments/83-t2-enface-convention/`): convención de
  ejes correcta (`transpose__flip_u__flip_v`), instrumento correlaciona
  (ratio 3.0-3.6×, `gate_pass=true`), vasculatura no (32 ratios en
  [0.81,1.04]). **Confirmado aquí** con máscaras GT completas y overlay visual
  directo (secciones 1 y 2), no solo con keypoints puntuales.
- **T2-89** (`experiments/89-t2-instrument-landmark/`): oráculo GT-a-GT de
  Procrustes/Umeyama sobre el subconjunto de instrumento (127/1214 casos con
  ≥2 puntos) — NO-GO por 3 órdenes de magnitud (AUC 0.0004 extrapolado,
  error medio 935px). Documentaron y corrigieron ellos mismos un bug real de
  `fit_closed_form_similarity` (ambigüedad de reflexión con 2 puntos) antes
  de reportar — buena disciplina, no se encontró ningún bug adicional en
  `run_task2_instrument_gate.py` al releerlo aquí línea por línea. **La
  sección 1.2 de este informe confirma visualmente por qué**: cuando hay
  correspondencia, es local y de calidad variable (algunos casos "pegan" muy
  bien, como el aquí mostrado; T2-89 documenta casos con error de cientos de
  px), y un solver de 2-3 puntos es numéricamente frágil ante esa varianza.
- **T2-82** (CNN de descriptores comunes): rechazado por control de
  OCT-shuffle (el loss no se degrada al barajar el volumen entre casos —
  `RESULTS.md:237`). Consistente con que ni vasculatura ni intensidad-cruda
  den señal densa explotable por un modelo de apariencia genérico.

---

## 6. Lo que esta auditoría NO resuelve

**No encontramos el error que el encargo pedía encontrar.** La cadena de
medición — geometría, convención, matriz GT, oráculos — pasa cada prueba que
se le aplicó, incluida una prueba de auto-consistencia sintética diseñada
específicamente para detectar un oráculo roto. Eso dificulta pero no cierra
la pregunta de fondo: **¿cómo llega el líder del leaderboard a 0.475?**

Lo que queda por descartar, en orden de esfuerzo creciente:

1. **Combinación de señales débiles.** Ninguna vía cerrada se probó en
   combinación (instrumento + prior de escala + prior de traslación
   poblacional, por ejemplo). T2-89 mide el techo de instrumento *solo*, no
   instrumento *más* algo más. Un modelo que solo necesita "no estar
   catastróficamente mal" en el 90% de casos sin instrumento (vía un prior
   razonable) y acertar bien en el 10% con instrumento podría, en teoría,
   sumar más AUC que cualquier vía aislada — pero el propio T2-89 muestra que
   ni siquiera el techo GT-a-GT de instrumento solo despega (0.0004), así que
   esto requeriría que el prior poblacional (ya con evidencia en contra:
   T2-81/T2-86, la escala no transfiere fuera de distribución) cargue con
   casi todo el peso. No verificado aquí.
2. **Un canal de señal que ninguna de las 7 vías cerradas modela.** Ejemplos
   no descartados: reflectividad/textura de fondo (más allá de vaso e
   instrumento), la forma del borde del disco óptico, o metadata de
   `properties.json` no explotada. `experiments/90-t2-literature-round2/`
   (revisión de literatura, no medición) lista candidatos de este tipo bajo
   "Frente A-D" pero no los prueba empíricamente contra los datos locales.
3. **Composición de mean+slab de intensidad cruda con un modelo que SÍ
   aprenda a ignorar el confusor de instrumento/viñeteado** en vez de
   quedarse pegado a él. El NCC de intensidad cruda (sección 2.1, hasta 0.59
   en el mejor slab de un caso) es positivo y no nulo en varios casos — no se
   descartó del todo que un modelo con la arquitectura/receptive-field
   correctos (no la que usó T2-82) pudiera extraer algo de esa señal parcial,
   aunque el control de OCT-shuffle de T2-82 pesa en contra.
4. **Diferencias en el propio proceso de scoring/submission de otros
   equipos** que no tengan que ver con señal cross-modal en absoluto —
   p.ej. si el corner-AUC del scorer premia desproporcionadamente casos
   "fáciles" por composición de escenario del test oculto, o si hay algún
   comportamiento del ingestion/scoring que un submission distinto exploque
   sin que sea "aprender del OCT" en el sentido que hemos estado midiendo.
   Completamente especulativo — no hay forma de verificarlo sin acceso al
   código de los otros equipos.

Ninguna de estas cuatro se puede descartar ni confirmar sin trabajo adicional
(medición 1 y 3 son baratas, CPU-only; 2 requiere literatura + medición; 4 no
es verificable localmente).

---

## Archivos de este rung

- `experiments/94-t2-upstream-bug/render_inspection.py` — script de renders,
  reutilizable (`process_case(scenario, frame)`).
- `experiments/94-t2-upstream-bug/renders/` — 5 grids completos por caso +
  `FIG_A_vessel_comparison.png` + `_overlay_vessel_S03_00040.png` +
  `_overlay_instrument_S01_00030.png`.
- `experiments/94-t2-upstream-bug/extracted/` — datos crudos extraídos
  selectivamente de los zips de train (Scenario_01 y 03, 5 frames) — se
  puede borrar sin pérdida, es reproducible desde los zips + el script.

## Lo que NO se hizo

- No se corrió nada en GPU ni en el pod.
- No se entrenó ningún modelo nuevo; toda la evidencia es oráculo/medición
  directa sobre anotaciones y segmentaciones GT.
- Solo 5 casos de train (2 escenarios) se inspeccionaron visualmente en
  detalle — suficiente para resolver la contradicción del encargo (que era
  sobre lo que se *ve*), insuficiente para generalizar los números de IoU/NCC
  a los 1214 casos con la misma confianza que los oráculos ya existentes
  (966-1214 casos). No se contradice nada de lo ya medido a escala completa.
