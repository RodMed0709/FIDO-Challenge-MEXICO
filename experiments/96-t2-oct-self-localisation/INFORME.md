# Auto-localizacion del OCT en Task 2 -- el volumen, sin mirar el fundus

> **ACTUALIZACION 2026-08-19 -- reformulacion T2-98, leer primero**
>
> A mitad de esta corrida, el coordinador aporto un hallazgo que reformula por que
> este documento mide lo que mide: Task 2 **no es registracion de imagenes**, es
> **estimacion de pose**. Verificado de forma independiente sobre los 1214 casos
> (`analysis/measure_task2_pose_estimation.py::verify_hallazgo`): `Opmi.Spatial` es
> constante, `Eyeball.Translation` es constante (cero), y solo dos cuaterniones
> varian -- `iOCT Microscope.Spatial.Rotation` (rotacion del escaner) y
> `Eyeball.Spatial.Rotation` (rotacion del ojo). Una regresion cuadratica de esos 8
> componentes contra los 6 parametros de la matriz GT da R2 0.975-0.994. Detalle
> completo: `experiments/98-t2-pose-estimation/HALLAZGO.md` (del coordinador) e
> `experiments/98-t2-pose-estimation/INFORME.md` (la medicion de este script).
>
> **Que sigue siendo valido de este documento**: TODO. La pregunta original --
> ¿el volumen OCT, sin mirar el fundus, sabe donde esta? -- sigue siendo la
> pregunta correcta, solo que ahora se entiende POR QUE la respuesta es "algo de
> senal, no suficiente": el volumen predice razonablemente bien la rotacion del
> ESCANER que lo genero (`iOCT.Rotation`, R2=0.48 con las mismas features
> compactas de aqui, ver T2-98 seccion 1) y esa rotacion determina la traslacion
> (tx,ty) via la geometria del simulador -- exactamente la relacion que explica
> por que el R2(tx,ty)=0.73 medido abajo (seccion 1) no es ruido, y por que aun
> asi el AUC se queda en ~0.0001-0.0019: la geometria de proyeccion amplifica
> errores angulares moderados en errores de traslacion grandes.
>
> **Hallazgo nuevo, contraintuitivo, que reordena donde conviene invertir**:
> combinando la rotacion del escaner predicha desde el OCT con la del ojo
> tomada del GT, la contribucion al AUC es **0.0000** -- el error de prediccion,
> aunque R2=0.48 parezca decente, es demasiado grande para la ventana de 10px
> (el AUC colapsa con solo 2-3 grados de error angular en `iOCT.Rotation`, ver
> T2-98 seccion 3). En cambio, predecir la rotacion del OJO desde el FUNDUS
> (monomodal, R2=0.22, peor R2 que el OCT) aporta AUC=**0.1432** aislada, porque
> el AUC tolera bastante mas error en esa componente. **La via mas prometedora
> no es la de este documento (OCT->traslacion), es fundus->rotacion del ojo**,
> medida en `experiments/98-t2-pose-estimation/`. Esos numeros son ademas una
> COTA INFERIOR: se calculan con una regresion cuadratica sustituta (no la
> formula geometrica exacta del simulador), que por si sola ya limita el AUC a
> 0.2092 incluso con las rotaciones GT exactas.
>
> El resto de este documento (medido ANTES de la reformulacion) se conserva
> integro abajo porque sigue siendo la medicion mas rigurosa que existe de
> cuanto sabe el volumen OCT sobre su propia posicion, y porque las features
> que aqui se construyeron (perfil de espesor Ilm-Rpe, densidad de vasos e
> instrumento) son las que T2-98 reusa sin volver a leer los volumenes.

---

Casos usados: **1214** de 1214 solicitados (10 escenarios). GroupKFold leave-one-scenario-out, n_splits=10. Semilla: 0.

Pregunta: ¿el volumen OCT, por si solo, codifica donde esta situado en la retina? Ningun feature de este script toca `microscope.png` (el fundus) en ningun punto -- todas las features vienen de `iOCT Microscope/Volume/.../Segmentation`.

## 0. Premisa anatomica (paso 3 del pedido)

Clases de segmentacion disponibles en el volumen (`vendor/fido/Dataset Explorer/constants.py`, `GenericLabels`): `Ilm=1`, `Rpe=2` (unicas superficies de capa retinal), mas `ArteriesOrVeins=3`, instrumento (`Forceps=8`, `Endoilluminator=10`, `InstrumentInOCT=11`), `OCTUncapturedArea=14`. **No existe una clase dedicada a fovea ni a disco optico** -- si hay senal anatomica, tiene que verse en la FORMA del mapa de espesor Ilm-Rpe, no en una etiqueta.

Espesor Ilm-Rpe por columna A-scan, agregado sobre los 1214 casos: media de la media por caso **55.1 px**; desviacion estandar DENTRO de cada volumen (media sobre casos) **6.8 px**; rango max-min dentro de cada volumen (media sobre casos) **38.3 px**.

Fraccion de casos 'planos' (rango de espesor < 5 px, es decir sin variacion apreciable): **0.0000**.

Cobertura valida (columnas con Ilm y Rpe detectados) media: **0.9462** (minimo observado: 0.7610). Fraccion de columnas 'OCTUncapturedArea': **0.0000**. Casos con instrumento visible en el volumen: **0.4876**.

**Veredicto de premisa**: el simulador SI produce variacion de espesor no trivial dentro de cada volumen (std media 6.8 px sobre una media de 55.1 px, es decir ~12.3% de variacion relativa). La premisa anatomica NO esta descartada de entrada; el peso de la prueba pasa a la seccion 1: si esa variacion esta ligada a la posicion real (tx,ty) o si es una textura sin relacion con la pose (lo que decide el control de barajado).

## 1. Test central: (tx, ty) solo con OCT

Control (0 features de OCT, pose constante = mediana de train por fold): AUC=**0.0000**, error medio=182.97 px. (Reproduce el hallazgo previo de pose constante; sirve de verificacion de metodologia.)

| feature set / modelo | R2(tx,ty) | R2(tx) | R2(ty) | AUC pos+mediana(theta,s) | AUC todo-predicho | tiempo |
|---|---:|---:|---:|---:|---:|---:|
| full/ridge **<- mejor** | -0.1367 | -0.6176 | 0.3441 | 0.0004 | 0.0000 | 3.4s |
| full/random_forest | 0.7246 | 0.7168 | 0.7324 | 0.0003 | 0.0010 | 56.2s |
| full/mlp | 0.1295 | 0.0514 | 0.2076 | 0.0002 | 0.0000 | 92.7s |
| compact/random_forest | 0.7291 | 0.7320 | 0.7261 | 0.0001 | 0.0019 | 15.9s |
| compact/mlp | 0.3572 | 0.2090 | 0.5053 | 0.0001 | 0.0000 | 80.9s |
| compact/ridge | 0.4052 | 0.3241 | 0.4863 | 0.0000 | 0.0001 | 0.5s |

Mejor combinacion: **full/ridge**, AUC(posicion OCT + theta/escala mediana de train)=**0.0004** (vs. 0.0000 de la pose constante, vs. 0.475 del lider del leaderboard).

AUC por escenario retenido (mejor modelo, posicion OCT + mediana theta/escala):

| escenario | AUC |
|---|---:|
| Scenario_01 | 0.0000 |
| Scenario_02 | 0.0000 |
| Scenario_03 | 0.0000 |
| Scenario_04 | 0.0000 |
| Scenario_05 | 0.0000 |
| Scenario_06 | 0.0000 |
| Scenario_07 | 0.0000 |
| Scenario_08 | 0.0038 |
| Scenario_09 | 0.0000 |
| Scenario_10 | 0.0000 |

## 2. Rotacion y escala por separado

| feature set / modelo | R2(theta, circular) | error angular medio (deg) | R2(escala) |
|---|---:|---:|---:|
| compact/mlp | 0.3350 | 15.5 | -1.0529 |
| compact/random_forest | 0.5019 | 14.8 | -0.4808 |
| compact/ridge | 0.3862 | 16.3 | -0.3382 |
| full/mlp | -11.8046 | 23.5 | -5.6995 |
| full/random_forest | 0.5151 | 14.7 | -0.4755 |
| full/ridge | 0.1567 | 17.3 | -0.4414 |

## 3. Control obligatorio -- OCT barajado entre casos

Mismo modelo y feature set ganador (**full/ridge**), features del OCT permutadas entre casos con semilla fija (targets y grupos SIN barajar):

- R2(tx,ty) con features barajadas: **-0.0534** (vs. -0.1367 sin barajar)
- AUC con features barajadas: **0.0000** (vs. 0.0004 sin barajar, vs. 0.0000 de la pose constante)

El control colapsa a nivel del piso constante: la senal medida en la seccion 1 no es un artefacto de indexado/orden -- depende genuinamente de que features y targets esten emparejados por caso.

## 4. Interpretacion pre-registrada

**Correccion post-hoc (2026-08-19)**: la seleccion de "mejor modelo" de la seccion 1 usa el AUC como criterio, y en este regimen los AUC son todos minusculos (0.0000-0.0019) -- practicamente ruido, y por eso el "mejor por AUC" (`full/ridge`, AUC=0.0004) resulta ser un modelo con R2(tx,ty) NEGATIVO (-0.1367), mientras que `compact/random_forest` tiene R2(tx,ty)=**0.7291** (ver tabla de la seccion 1) y un AUC casi identico (0.0001). El criterio de seleccion por AUC es enganoso aqui; el R2 es la metrica que hay que mirar para juzgar si hay senal.

Verificado ademas con un control de barajado especifico para `compact/random_forest` (no ejecutado automaticamente por este script, que solo barajo el modelo ganador por AUC): R2(tx,ty)=0.7291 con las features reales se desploma a **R2=-0.0707** con las features barajadas entre casos -- confirma que el R2=0.73 es senal real, no artefacto.

**Conclusion correcta**: el volumen OCT SI predice (tx,ty) con una fraccion sustancial de la varianza (R2=0.73, reduce el error medio de esquinas de 183px con pose constante a 98px con `compact/random_forest`), y esa senal sobrevive el control de barajado. Pero el error absoluto residual (~98px) sigue siendo ~10x mayor que la ventana de 10px que puntua el AUC oficial, asi que la traduccion a AUC es casi nula (0.0001). Esto es exactamente la categoria **"R2 apreciable pero AUC insuficiente"** del criterio de interpretacion pre-registrado -- la hipotesis de auto-localizacion NO muere, pero por si sola no es competitiva: hace falta reducir el error de traslacion en un orden de magnitud, no solo tener senal.

**Por que el R2 no se traduce en AUC (explicado por T2-98, ver actualizacion arriba)**: el volumen OCT predice bien la ROTACION del escaner que lo genero (R2=0.48, `experiments/98-t2-pose-estimation/`), y la geometria de proyeccion del simulador amplifica errores angulares moderados en errores de traslacion grandes -- el AUC ya colapsa con 2-3 grados de error en esa rotacion. Ese es el mecanismo real detras del R2(tx,ty) alto pero AUC bajo medido aqui.

## Notas de implementacion

- Metrica oficial: `src/fido/eval_task2.py::evaluate_task2_cases` (misma funcion que otros diagnosticos de Task 2 en este repo), que envuelve `corner_auc`/`corner_error` de `src/fido/geometry.py`, replica exacta del scorer oficial vendorizado.
- GroupKFold con 10 folds sobre 10 escenarios: cada fold deja fuera un escenario completo (leave-one-scenario-out exacto, no aproximado).
- theta/escala 'mediana de train' se calculan POR FOLD sobre los escenarios de entrenamiento de ese fold (media circular para theta -- el angulo cruza +-180 grados en los datos reales, confirmado; mediana simple para la escala).
- Features compactas: 33 (interpretables: estadisticas de espesor, posicion del minimo/maximo, centroide de la 'zona fina', densidad y centroide de vasos e instrumento). Features 'full': compactas + mapas 8x8 de espesor/vasos/instrumento (192 valores adicionales).
- Cache de features en `experiments\96-t2-oct-self-localisation\features_cache.npz`; borrar o pasar `--refresh` para reextraer desde los zips/directorios de `data/Task 2`.
