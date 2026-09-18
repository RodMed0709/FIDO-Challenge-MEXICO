# T2-101 — Techo honesto end-to-end de la via geometrica, SIN GT alguno

**Fecha**: 2026-08-19. Script reproducible:
`analysis/measure_task2_honest_ceiling.py`
(`KMP_DUPLICATE_LIB_OK=TRUE python analysis/measure_task2_honest_ceiling.py`;
numpy/sklearn, local, CPU, ~5s -- reusa los caches de features ya extraidos
por T2-96 y T2-100, no relee imagenes). Resultados crudos en
`experiments/101-t2-honest-ceiling/results.json`.

**Pregunta pre-registrada** (mandato del coordinador, previa a este script):
la via geometrica de Task 2 (`analysis/derive_task2_geometry.py`, T2-98/99) da
AUC LOSO=0.1512 con coeficientes globales, y T2-100 mejoro a AUC=0.3201
recalibrando los 8 coeficientes de escala/traslacion por escenario desde una
feature del fundus (`profile_edge_r50`). **Pero ambos numeros usan
`iOCT.Rotation` LEIDO DEL GT** para calcular `theta`, `gx`, `gy` -- ese campo
no existe en `inference(task_id, oct_volume, opmi_image, model)`, solo hay
`oct_volume` y `opmi_image`. **¿Cual es el AUC LOSO real si se reemplaza esa
rotacion GT por una prediccion entrenada solo con el volumen OCT?** Ese es el
unico numero que podria ir a Codabench. Criterio pre-registrado: si sale
< 0.05, la via no es desplegable y no se construye submission.

## Resultado headline

| variante | AUC LOSO | mean err (px) | fuente de `iOCT.Rotation` | fuente de coef. escala/tx/ty |
|---|---:|---:|---|---|
| techo T2-98 (coef. globales, rotacion GT) | 0.1512 | 16.85 | GT | globales (sin recalibrar) |
| techo T2-100 (coef. por-escenario desde fundus, rotacion GT) | 0.3201 | 11.14 | **GT** | fundus (`profile_edge_r50`) |
| rotacion PREDICHA (OCT) + coef. globales | 0.0000 | 123.28 | volumen OCT (OOF) | globales |
| **rotacion PREDICHA (OCT) + coef. PREDICHOS (fundus) -- HONESTO, sin GT alguno** | **0.0004** | **122.82** | **volumen OCT (OOF)** | **fundus (`profile_edge_r50`)** |

La fila 2 (0.3201) se reprodujo en este mismo script como ancla de
consistencia (`run_reference_gt_rotation_pred_coefs`, coincide exacto con
T2-100) antes de medir la fila 4 -- confirma que el pipeline de este script es
el mismo modelo geometrico, no una reimplementacion divergente.

**El AUC honesto es 0.0004, indistinguible de cero.** Confirma el techo duro
que ya anticipaba el barrido de sensibilidad de `GEOMETRIA_EXACTA.md` §5: el
AUC de este modelo colapsa a ~0 con solo 2-5° de error angular en
`iOCT.Rotation`. El error angular REAL de la mejor prediccion disponible desde
el volumen OCT (features compactas de T2-96, `RandomForestRegressor`, mismo
modelo que dio R2=0.4834 en T2-98 §1) es **mediana 14.30°, media 16.86°** --
30-150x mayor que el umbral de colapso. No es un margen estrecho: es una
brecha de casi dos ordenes de magnitud.

## Metodologia (nested LOSO real, sin fuga)

Por cada uno de los 10 folds (un escenario fuera):

1. **Rotacion**: se entrena `RandomForestRegressor(n_estimators=300,
   min_samples_leaf=3)` sobre las 33 features compactas del volumen OCT
   (T2-96: perfil de espesor Ilm-Rpe, densidad de vasos/instrumento) de los 9
   escenarios de train, prediciendo los 4 componentes de `iOCT.Rotation`. Se
   predice (OOF real, nunca visto en train) para el escenario de test y se
   normaliza a cuaternion unitario.
2. **Boresight + theta0** (`dtg.calibrate_boresight`, `dtg.fit_model`):
   ajustados usando SOLO `iOCT.Rotation` GT de los 9 escenarios de train (son
   constantes del rig fisico, legitimo usarlas -- no dependen del caso de
   test).
3. **Coeficientes oraculo por escenario de TRAIN** (`s0,s1,c_gx_tx,c_gy_tx,
   c0_tx,c_gx_ty,c_gy_ty,c0_ty`): un ajuste por cada uno de los 9 escenarios
   de train, usando su propio GT (nunca el del escenario de test).
4. **Predictor imagen->coeficientes**: regresion lineal (`profile_edge_r50`,
   1 feature, 2 parametros por objetivo, igual que T2-100) ajustada sobre los
   9 puntos (medias por escenario de train), prediciendo los 8 coeficientes
   del escenario de test a partir de SU media de `profile_edge_r50`.
5. **Composicion final**: `dtg.predict_matrix(q_pred_test, v0, e1, e2,
   params_pred)` -- **la rotacion usada aqui es la PREDICHA del paso 1, no la
   GT**. Esta es la unica diferencia real respecto a T2-100 (que usaba
   `ioct_q[test_idx]`, GT, en este mismo paso).
6. `corner_error`/`corner_auc` oficiales (`src/fido/eval_task2.py`,
   `src/fido/geometry.py`) sobre los errores agrupados de los 10 folds.

Ningun paso lee GT del escenario de test en ningun punto. Es el numero mas
honesto que se puede calcular con los datos de train disponibles, y
corresponde exactamente a lo que `inference()` podria hacer con
`oct_volume` + `opmi_image` reales.

## Interpretacion

El cuello de botella no es el modelo geometrico (T2-98 ya lo deja en un AUC
razonable, 0.1512-0.3201, cuando la rotacion es exacta) ni el predictor de
coeficientes desde el fundus (T2-100 ya demostro que generaliza, percentil
98.8 contra ruido puro). **El cuello es, exactamente como preveia el mandato
del coordinador, la precision angular de `iOCT.Rotation` estimada desde el
volumen**: el mejor modelo disponible (R2=0.4834, verificado con control de
barajado en T2-96/98) tiene un error angular real de ~14-17° -- la metrica
oficial exige sub-0.1-0.5° para producir AUC apreciable con este modelo de
composicion. No es una cuestion de ajustar hiperparametros o probar otro
regresor barato: la brecha es de casi dos ordenes de magnitud, y no hay
indicio en T2-96/98 (que ya probo ridge, random forest, MLP, features
compactas y "full" con mapas 8x8) de que exista una feature barata del
volumen capaz de cerrarla.

## Veredicto frente al criterio pre-registrado

| criterio | umbral | resultado (0.0004) |
|---|---:|---|
| via no desplegable, no construir submission | AUC < 0.05 | **se cumple** -- 0.0004 << 0.05 |
| via apreciable, justifica submission | AUC >= 0.05 | no se alcanza |

**Conclusion**: la via geometrica de Task 2, evaluada end-to-end sin ningun
GT (ni `iOCT.Rotation` ni coeficientes de escenario), da un AUC LOSO
indistinguible de cero (0.0004). Los numeros de 0.1512/0.7204/0.3201
reportados en T2-98/99/100 son techos condicionados a informacion que no
existe en `inference()` real (`oct_volume`, `opmi_image` solamente) y **no
deben citarse como el score esperable de una submission** -- son cotas
superiores de "cuanto vale la formula si la rotacion se supiera", no
resultados desplegables. Siguiendo el mandato pre-registrado, **no se
construye la submission `submissions/r07-task2-geometry/`**: haria falta
reducir el error angular de `iOCT.Rotation` estimado desde el volumen OCT en
casi dos ordenes de magnitud (de ~15° a <0.5°) antes de que esta via aporte
nada por encima del 0.0000 actual en Codabench.

## Que queda abierto (no resuelto aqui, fuera del alcance de este mandato)

- No se probaron arquitecturas mas caras para estimar `iOCT.Rotation` (CNN
  sobre el volumen 3D bruto en vez de features tabulares baratas) -- T2-96/98
  ya usaron random forest/MLP/ridge sobre features compactas e interpretables
  y "full" (+ mapas 8x8), sin indicio de que una red profunda fuera a cerrar
  una brecha de 30-150x en precision angular con los mismos ~1090 casos de
  entrenamiento por fold, pero no es una prueba formal de que sea imposible.
- Los experimentos 96-100 (auto-localizacion OCT, estimacion de pose,
  heterogeneidad de escenario, coeficientes desde imagen) no estan todavia
  registrados como peldanos en `ATTACK_LADDER.md` -- son anteriores a este
  mandato y quedan fuera de su alcance; se deja senalado para que se
  incorporen en una pasada separada.
