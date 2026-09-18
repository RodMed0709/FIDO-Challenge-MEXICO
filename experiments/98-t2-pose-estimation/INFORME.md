# Task 2 como estimacion de pose (T2-98) -- iOCT.Rotation y Eyeball.Rotation

Reformulacion del coordinador, verificada de forma INDEPENDIENTE en este script (`verify_hallazgo`) sobre los 1214 casos de train: `Opmi.Spatial` es constante, `Eyeball.Translation` es constante (cero), y solo dos cuaterniones varian.

| campo | std por componente |
|---|---|
| `iOCT.Rotation` | [0.0583, 0.0917, 0.0592, 0.2124] |
| `Eyeball.Rotation` | [0.0297, 0.0035, 0.0106, 0.6741] |
| `Eyeball.Translation` | [0.0, 0.0, 0.0] (constante, cero) |
| `Opmi.Rotation` | [0.0, 0.0, 0.0, 0.0] (constante) |

Regresion cuadratica de los 8 componentes de ambos cuaterniones contra los 6 parametros de la matriz GT (ajustada sobre los 1214 casos, como aproximacion de la geometria fija del simulador, NO la formula exacta -- ver limitacion abajo): R2 por parametro = [0.976, 0.975, 0.994, 0.975, 0.976, 0.994], MAE por parametro = [5.74, 5.71, 6.65, 5.74, 5.69, 7.02] px.

**Limitacion central de este informe**: la regresion sustituta NO es la formula geometrica exacta (proyeccion 3D real del simulador con los cuaterniones e intrinsecos de camara) -- es una aproximacion polinomica. Eso le pone un TECHO DURO a todo lo que sigue: incluso alimentandola con los cuaterniones GT exactos (sin ningun error de prediccion), el AUC que produce es **0.2092**, no 1.0. Todos los numeros de AUC de este informe deben leerse como una COTA INFERIOR de lo que la via de estimacion de pose podria lograr con la composicion geometrica exacta -- no como el techo real de la via.

## 1. iOCT.Rotation desde el VOLUMEN OCT (features reusadas de T2-96)

Mismas features del volumen (perfil de espesor Ilm-Rpe, densidad de vasos e instrumento, compactas + grid 8x8) que en `experiments/96-t2-oct-self-localisation/`, solo cambia el target: de (tx,ty) a los 4 componentes del cuaternion `iOCT.Rotation`.

| feature set / modelo | R2 conjunto | R2 por componente [x,y,z,w] | tiempo |
|---|---:|---|---:|
| compact/random_forest **<- mejor** | 0.4834 | [0.4648, 0.5063, 0.4464, 0.5163] | 5.3s |
| full/random_forest | 0.4752 | [0.4229, 0.5203, 0.4316, 0.5259] | 16.4s |
| compact/ridge | 0.4184 | [0.3965, 0.3964, 0.4739, 0.4069] | 0.1s |
| full/ridge | 0.2105 | [0.3096, 0.1194, 0.2328, 0.18] | 0.8s |
| compact/mlp | 0.1008 | [0.207, -0.0597, -0.0635, 0.3195] | 23.8s |
| full/mlp | -44.5190 | [-140.3985, -13.882, -5.5474, -18.2482] | 66.2s |

Control de barajado (mejor modelo, `compact/random_forest`): R2 con features de OCT barajadas entre casos = **-0.0385** (vs. 0.4834 sin barajar).

## 2. Eyeball.Rotation desde el FUNDUS (features nuevas, monomodal)

Features nuevas del lado microscopio (nunca del OCT): mascara de retina visible (`ilm.png`, area/centroide/orientacion), mascara de vasos (`arteriesorveins.png`, densidad/centroide/orientacion), y brillo (proxy del disco optico -- percentil 95 de intensidad dentro de la retina visible). Compactas + grid 8x8.

| feature set / modelo | R2 conjunto | R2 por componente [x,y,z,w] | tiempo |
|---|---:|---|---:|
| compact/random_forest **<- mejor** | 0.2218 | [0.4628, -0.0803, 0.0752, 0.4296] | 5.4s |
| full/random_forest | 0.1376 | [0.2407, -0.1607, 0.1779, 0.2924] | 34.3s |
| compact/ridge | 0.0170 | [0.642, -0.0586, -0.7323, 0.217] | 0.0s |
| full/ridge | -0.3888 | [0.5539, -0.3954, -1.5088, -0.2051] | 0.8s |
| compact/mlp | -823.0215 | [-35.7514, -3010.5538, -245.8127, 0.032] | 11.8s |
| full/mlp | -2505.1862 | [-116.6906, -8613.0079, -1290.2351, -0.8111] | 9.9s |

Control de barajado (mejor modelo, `compact/random_forest`): R2 con features de fundus barajadas entre casos = **-0.1097** (vs. 0.2218 sin barajar).

## Contribucion al AUC oficial (via la regresion sustituta, sujeta a la limitacion de arriba)

| combinacion | AUC | error medio (px) | error mediano (px) |
|---|---:|---:|---:|
| gt_both (sanity: reproduce la matriz real via el sustituto, sin prediccion) | 0.2092 | 12.72 | 9.19 |
| pred_ioct + gt_eye (contribucion del volumen OCT sola) | 0.0000 | 123.01 | 109.24 |
| gt_ioct + pred_eye (contribucion del fundus solo) | 0.1432 | 29.81 | 12.06 |
| pred_ioct + pred_eye (todo predicho, ambas modalidades) | 0.0000 | 137.09 | 109.24 |

## 3. Cuanta precision angular hace falta (barrido, sujeto a la misma limitacion)

Techo con cuaterniones GT exactos (angulo=0): AUC=**0.2092**. A partir de ahi se perturba cada cuaternion con ruido angular de magnitud creciente (eje aleatorio, 5 tiradas por magnitud) y se mide cuanto tarda en desplomarse el AUC.

| angulo (deg) | AUC ambos perturbados | AUC solo iOCT | AUC solo ojo |
|---:|---:|---:|---:|
| 0 | 0.2092 | 0.2092 | 0.2092 |
| 0.5 | 0.0892 | 0.1059 | 0.1651 |
| 1 | 0.0182 | 0.0284 | 0.1118 |
| 2 | 0.0018 | 0.0056 | 0.0493 |
| 3 | 0.0008 | 0.0018 | 0.0266 |
| 5 | 0.0000 | 0.0000 | 0.0087 |
| 7.5 | 0.0000 | 0.0000 | 0.0038 |
| 10 | 0.0000 | 0.0000 | 0.0015 |
| 15 | 0.0000 | 0.0000 | 0.0002 |
| 20 | 0.0000 | 0.0000 | 0.0002 |
| 30 | 0.0000 | 0.0000 | 0.0000 |
| 45 | 0.0000 | 0.0000 | 0.0000 |

`iOCT.Rotation` es mucho mas sensible que `Eyeball.Rotation`: perturbar solo el cuaternion del escaner desploma el AUC en 2-3 grados, mientras que perturbar solo el del ojo tolera bastante mas antes de colapsar. La rotacion del escaner es el eslabon critico: define que ventana de la retina capturo el volumen, y un pequeno error ahi mueve la traslacion completa fuera de la ventana de 10px que puntua la metrica.

## Interpretacion

AUC con AMBAS rotaciones predichas (OOF) = **0.0000**. Contribucion aislada: solo iOCT predicho (ojo en GT) = **0.0000**; solo ojo predicho (iOCT en GT) = **0.1432**. Techo del sustituto con ambas rotaciones GT (sin ninguna prediccion) = 0.2092.

**Asimetria real y contraintuitiva**: `iOCT.Rotation` tiene MEJOR R2 (seccion 1, 0.4834) que `Eyeball.Rotation` (seccion 2, 0.2218), pero su contribucion aislada al AUC es MENOR. Se explica por la seccion 3: el AUC es mucho mas sensible a errores en `iOCT.Rotation` (colapsa en 2-3 grados) que a errores en `Eyeball.Rotation` (tolera bastante mas), asi que el mismo nivel de error de prediccion pesa mucho mas cuando cae sobre la rotacion del escaner. La via del FUNDUS (monomodal, mas facil de mejorar) es la que mas cerca esta de aportar algo al AUC, no la del volumen OCT pese a su R2 mas alto.

Mejor contribucion aislada AUC=0.1432 >= 0.10 (recordar que es COTA INFERIOR por la limitacion del sustituto): la via de pose es viable y vale la pena invertir en derivar la composicion geometrica exacta.

## Pendientes declarados (de `HALLAZGO.md`, no resueltos aqui)

- Derivar la composicion geometrica EXACTA (proyeccion 3D del simulador con los cuaterniones e intrinsecos de camara reales) en vez de la regresion cuadratica sustituta -- resolveria la limitacion central de este informe y probablemente subiria el techo bastante por encima de 0.2092.
- Confirmar si Task 1 anota `eye pose` con el mismo convenio (50x mas frames disponibles para supervisar `Eyeball.Rotation` desde el fundus, si es cierto).

## Notas de implementacion

- Reusa el cache de features OCT de `measure_task2_oct_self_localisation.py` (`experiments\96-t2-oct-self-localisation\features_cache.npz`) -- no se releyeron los volumenes.
- Features de fundus nuevas, cacheadas en `experiments\98-t2-pose-estimation\fundus_features_cache.npz`.
- GroupKFold leave-one-scenario-out, n_splits=10, igual que en T2-96.
- `evaluate_task2_cases`/`corner_auc`/`corner_error` de `src/fido/eval_task2.py` y `src/fido/geometry.py` -- misma metrica oficial que el resto del proyecto.
