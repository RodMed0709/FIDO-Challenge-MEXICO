# RESULTS — qué funcionó, qué no, y por qué

> Un renglón por corrida con número. Los fracasos se registran igual que los
> éxitos: un peldaño sin resultado escrito cuenta como no ejecutado
> (`CONSTITUTION.md` §5).
>
> Regla de lectura: **ningún número entra aquí sin el comando y el commit que
> lo produjo.** Un número sin su corrida no es un resultado, es una anécdota.

---

## Leaderboard de referencia (fase Competition, 2026-08-14)

Lo que hay que superar. Consultado con
`https://www.codabench.org/api/phases/27554/get_leaderboard/`.

| Task 2 — registración | score | | Task 1 — keypoints | score |
|---|---|---|---|---|
| cnielsen | **0.475** | | kwonmj | **0.619** |
| Alaa Senjab | 0.296 | | Alaa Senjab | 0.618 |
| oli4more | 0.280 | | akkkkk | 0.604 |
| kwonmj | 0.279 | | cnielsen | 0.588 |
| dancies | 0.112 | | dancies | 0.565 |
| akkkkk | 0.111 | | campana | 0.549 |
| alexlin | 0.047 | | Alex | 0.479 |
| 4 equipos | 0.000 | | tianmaxingkong | 0.418 |

---

## Techos y pisos conocidos

Medidos, no estimados. Contexto para juzgar cualquier resultado futuro.

| Cota | Valor | De dónde sale |
|---|---|---|
| Task 2 — predictor constante | **0.000** | `analysis/verify_task2_structure.py` |
| Task 2 — techo de la parametrización de 4 DOF | **0.794** | residuo medio de 1.74 px al forzar la forma |
| Task 1 — mejor `distance_AUC` constante posible | **0.0445** | `analysis/explore_task1_gt.py` |
| Task 1 — aporte de esa constante al score final | **0.0134** | peso 0.3 |
| Task 1 — error del keypoint medio como predicción | 131 px | ídem |

**Cómo leer un resultado**: superar el piso trivial no es habilidad. Un score de
Task 1 por debajo de `0.7 × keypoint_AUC` con `distance_AUC ≈ 0.045` significa
que la componente de distancia sigue sin explotarse.

---

## Corridas

### Submissions reales a Codabench

| Submission | Task | Score local (Mock Test) | Score Codabench | Ejecución | Estado |
|---|---|---|---|---|---|
| `r01` — `submissions/r01-task1-keypoint/` | Task 1 | **0.4800** | **0.5386** (`keypoint_auc=0.7309`, `distance_auc=0.0900`) | 100 casos, 24.8 s total, 0.25 s/caso, GPU | puntuada |
| `r02-best` — `submissions/r02-best/` | Task 1 | **0.5945** (`keypoint_auc=0.818182`, `distance_auc=0.072727`) | — | no se subió | solo local |
| `r03` — `submissions/r03-dist-v2/` | Task 1 | **0.6000** (`keypoint_auc=0.818182`, `distance_auc=0.090909`) | **0.569** (`keypoint_auc`: pendiente; `distance_auc`: pendiente) | no medido | puntuada |

`inference.py` es idéntico byte a byte entre `r01` y `r03`. La comparación
verificada por hash de tensores muestra que cambian las dos cabezas de Task 1
(keypoint y distance); `model_1.pth` de Task 2 es idéntico entre ambas.

**Lección**: el harness local no predice Codabench. `r01` pasó de 0.4800 local
a 0.5386 real (el real fue optimista frente al local), mientras `r03` pasó de
0.6000 local a 0.569 real (el real fue pesimista frente al local). Con solo 5
casos de Mock Test, el harness sirve para verificar que el contenedor no
truena, no para elegir modelo.

### Entrenamientos y diagnósticos

| # | Fecha | Qué se probó | Score local | Score Codabench | Costo | Veredicto |
|---|---|---|---|---|---|---|
| R00 | 2026-08-14 | Submission de humo: contrato de entrega, sin modelo | T1 0.000000 / T2 0.000000 | — | $0 | ✅ **el contrato funciona**. Cero era lo esperado; lo que se probaba es que saliera un número en vez de un error de ingestión |
| R01 | 2026-08-16/17 | T1-R5 (UNet cánula+ILM + distancia geométrica), primer entrenamiento real sobre dataset completo (`train_task1_unet.py`, 5 folds, fold 0) | `val_distance_auc=0.5681` en época 2, 12,566 casos (piso trivial 0.0445 → ~12.8x mejor) | — (no es submission, es entrenamiento local) | ~$5h de pod ($0.99/h) | 🔄 **parcial** — corrida murió en época 3 por `OSError: [Errno 6]` (I/O transitorio de MooseFS, no bug de código). Checkpoint de época 2 se conservó (`checkpoints/task1_real/model_0.pth`). Relanzado con fix de reintentos (ver hallazgo abajo) |
| R02 | 2026-08-17 | T2-R4 Motor 1 (template matching clásico de vasos, `match_vessels`) contra el Mock Test, con máscara GT de vaso perfecta como entrada (aísla el método del segmentador) | `corner_error` media=573.69px (min 226.60, max 780.30), `corner_auc(0..10px)=0.0000` sobre 4/5 casos válidos (1 caso degenerado, `enface_vessel.std()==0`, saltado con aviso) | — (no es submission) | $0, CPU local, agente de debug (~108k tokens) | ❌ **REFUTADO por evidencia oráculo, no por bug**: caso sintético autoconsistente pasa (`corner_error=0.09px`), y con los parámetros REALES del GT (sin búsqueda) el NCC entre el patch alineado y el fundus real es ≈0 en los 4 casos — la señal `enface_vessel_density` no correlaciona con el vaso del fundus ni en la respuesta perfecta. Ver `ATTACK_LADDER.md` T2-R4 y `analysis/verify_vessel_template_match_synthetic.py` |

| R03 | 2026-08-17 | T2-R4 Motor 1, ronda 2: probar si otra representación de `enface_vessel_density` (binarizada, dilatada 2-21px, búsqueda libre de traslación) recupera señal | oráculo NCC 0.007-0.018 en posición GT real (vs 0.0071 de la señal actual); búsqueda libre sube a 0.24-0.31 pero en posición que NO coincide con el GT en 4/4 casos | — | $0, CPU local, agente (~141k tokens) | ❌ **cerrado, sin fix posible con transformación morfológica simple**. El score alto de la búsqueda libre es autosimilitud del árbol vascular (falso positivo), mismo modo de fallo que R02. `enface_vessel` derivado solo de la clase `ArteriesOrVeins` no alcanza para template matching clásico bajo ninguna variante probada. Detalle en `ATTACK_LADDER.md` T2-R4 y `analysis/verify_vessel_signal_candidates.py` |

| R04 | 2026-08-17 | T1-R3 (keypoint CNN + heatmap), primer smoke test de overfit (5 casos Mock Test, CPU local) | `keypoint_auc=0.145`, un ejemplo individual baja de >150px a 3.0px, loss 151.2→31.3 en 60 pasos | — | $0, CPU local, agente (~133k+109k tokens en 2 rondas) | ✅ **sin bugs de código** (a diferencia de T2-R2 y T1-R5, que tuvieron 4 y 2 respectivamente) — contrato de datos verificado (orden (x,y), fundus 1024×1024 real). Hallazgo real, no bug: norma de gradiente cruda explota (751→7469 en 40 pasos), `clip_grad_norm_` con `max_norm=1.0` hardcoded domina casi todos los pasos. Expuesto como `--max-grad-norm` (default 1.0) para poder subirlo si la convergencia sigue lenta con datos reales |

| R05 | 2026-08-17 | T2-R2 (heatmap correlación cruzada + regresión θ,s), entrenamiento completo real (25 épocas, fold 0/5, 966 train/248 val) | `val_corner_auc=0.0000` en las 5 épocas medidas; `val_mean_error` 203.29px (ep5) → **98.74px (ep25, mejor)** | — (no es submission) | ~$2/pod, 3 relanzamientos por fixes de rendimiento (ver hallazgos arriba) | 🔄 **aprende pero no cruza el umbral de AUC** (<10px) — error cae >50% de época 5 a 25 sin aplanarse del todo, `param_mse` converge sólido. No es bug (smoke test ya predijo ~165-200px con 5 ejemplos; 98.74px sobre 248 reales nunca vistos es mejor). Candidatos siguientes: más épocas, cascada T2-R6, revisar prior de escala. Detalle en `ATTACK_LADDER.md` T2-R2 |

**Cicatriz de resiliencia (2026-08-17)**: T1-R5 murió otra vez con `OSError: [Errno 6]` — esta vez agotando los 3 reintentos de 0.3s (el error persistió >1s, más largo que lo visto antes). Fix: `_retry_on_io_errors` en `common.py` subido a 6 intentos con backoff exponencial (0.5, 1, 2, 4, 8s en vez de 3×0.3s fijo). Además se aplicó a `task1.py` el mismo patrón defensivo que ya tenía `task2.py`: `find_task1_cases` salta escenarios/casos con `OSError` en vez de tronar el escaneo completo, y `Task1Dataset.__getitem__` cae al siguiente caso si uno falla tras agotar reintentos. Relanzado con `--val-every 3` (menor impacto que en T2 ya que Task1 carga solo 2 Bscans+2 segs por caso, no 128 slices, pero se agregó por consistencia y porque el costo de perder una corrida completa por un solo caso roto es alto). El fallback por-caso funcionó en producción: saltó `Scenario_07/05261` tras agotar reintentos, sin tronar la corrida.

**Cuarto hallazgo, mismo problema**: aun con los fixes, el cómputo de `class_weights` (pesos de clase por frecuencia inversa, necesario porque cánula es solo 0.008% de los píxeles) recorría el **train set completo** (~49k casos tras el split, 4 lecturas/caso) antes de que empezara la época 1 — más de una hora de I/O solo para eso. Fix: muestrear hasta 200 batches (1600 casos) en vez del dataset completo — la fracción de clase es una propiedad casi constante de la anatomía, no de la muestra, mismo razonamiento que ya usaba `train_fundus_vessel_seg.py` con 5 batches. Lección repetida: cualquier paso de "preparación" que itere el DataLoader completo antes del loop de entrenamiento paga el mismo costo de I/O que una época real — muestrear, no recorrer todo, salvo que el número en sí sea el resultado que se está midiendo.

**Quinto hallazgo, mismo problema**: incluso muestreando 200 batches, la corrida se quedó 90+ minutos sin arrancar la época 1 — MooseFS tuvo una sesión particularmente lenta (un caso concreto, `Scenario_07/05261`, agotó los 6 reintentos con backoff repetidamente). Fix definitivo: `--class-weights` acepta los pesos ya medidos en una corrida anterior real (`0.0341661,0.4904578,2.4753761`, de `class_counts=[24670209770, 119718495, 4699831]` — fondo/Ilm/cánula) y salta el escaneo por completo. La fracción de clase es una propiedad de la anatomía del dataset, no cambia entre corridas del mismo split — no hay necesidad real de remedirla cada vez que MooseFS coopera.

| R06 | 2026-08-17 | T2-R2 **con los 3 bugs estructurales corregidos** (centro→esquina, sesgo de media celda, plantilla cuadrada) + rebalanceo de pérdidas, 60 épocas | 110.97 (ép5) → 89.52 (ép15) → 80.24 (ép20) → 74.48 (ép25) → 57.81 (ép45) → **66.25px (ép60)**. `val_corner_auc` **0.0048** (mejor, ép60), primer valor no-cero del proyecto | — | ~$1.5 | ✅ **el techo estructural desapareció** — con localización perfecta el decodificador viejo daba AUC 0.0000 y el nuevo 0.7848, o sea la corrida anterior no podía puntuar aunque entrenara perfecto. Ahora sí puntúa, pero lejos del líder (0.475). El error se aplana ruidoso en 57-72px. Siguiente límite medido: la rotación sale de features mean-pooled (ver abajo) |
| R07 | 2026-08-18 | T2-80, diagnóstico fold 0 del checkpoint `task2_fixed`: sustitución independiente de componentes GT y control paired/OCT-shuffle | Baseline AUC **0.004765**, mediana **41.451px**. GT posición: **0.048021** / 19.673px; GT rotación: 0.011364 / 35.253px; GT escala: 0.018695 / 37.118px. M2 paired: **6.855%** a 10px, mediana **37.121px**; shuffle: **0.403%**, **107.296px**. M1 paired top-1%: **0.340%** (vecino), **0.391%** (bilineal) | — (diagnóstico local; no submission) | corrida ya materializada en pod | ✅ **T2-80 DONE**. Posición domina el error del baseline. M2 se degrada fuertemente con shuffle y demuestra señal OCT específica; M1 falla el gate de representación común (`>=50%`) por dos órdenes de magnitud. Abre T2-81, cuyo código está aprobado pero aún no fue entrenado. Artefactos: `experiments/80-t2-diagnostics/` |
| R08 | 2026-08-18 | T2-81, augmentación geométrica exacta de escala como único cambio, fold 0/5, seed 0 | Holdout sintético (496 evaluaciones = 248 casos x 2 escalas): AUC **0.000550 -> 0.006048**; error medio **85.765 -> 65.305px**; mediana **61.056 -> 42.100px**; scale MAE **31.163 -> 14.278px**, reducción relativa **54.184%**. Scenario_09 AUC 0.000745 -> 0.007700; Scenario_10 0 -> 0.001399. Mejor checkpoint de entrenamiento: época **38**, AUC **0.0092**, error medio **77.60px** sobre 248 casos | — (evaluación train-only; no submission) | corrida materializada en pod | ✅ **T2-81 DONE / GO**. Supera los tres gates: AUC sube, reducción de escala >20% y ninguna caída por escenario (`observed_max_per_scenario_auc_drop=-0.001399`). Rango derivado de 966 train, 248 val disjuntos; `selection_uses_mock=false`. Abre T2-82, ya en ejecución por builder. |

*Siguiente peldaño Task 2: T2-82 está IN_PROGRESS por builder.*

---

## Resultados medidos el 2026-08-18/19

### A/B de backbone: CNN desde cero contra DINOv2

| Task | Variante | Mejor resultado | Veredicto |
|---|---|---|---|
| Task 1 keypoint | CNN desde cero | `val_keypoint_auc=0.8537` | gana |
| Task 1 keypoint | DINOv2 | `val_keypoint_auc=0.8005` en época 6; `val_mean_error=1.72 px`; n=14399 | pierde |
| Task 2 | CNN desde cero | `val_corner_auc=0.0048`; `val_mean_error=66.25 px` | gana |
| Task 2 | DINOv2 | `val_corner_auc=0.0000` en épocas 30, 35 y 40; `val_mean_error` 103.25 → 125.17 → 136.60 px | divergió |

**Conclusión**: DINOv2 pierde en las dos tasks. No queda como candidato
principal ni como A/B pendiente.

### Task 1 — distancia v2

`/workspace/t1_dist_v2.log`: mejor `val_distance_auc=0.6113` en época 10,
`val_distance_mae=4.58 px`, `val_pixel_acc=0.9984`, n=12127.

**Advertencia obligatoria**: esa `evaluate()` descarta con `continue` los casos
sin gap medible, mientras el scorer oficial los penaliza. El 0.6113 está
inflado y no es comparable con el 0.0900 de Codabench.

### Task 2 — T2-R11/R11b, señal oráculo y convención en-face

T2-R11 (`analysis/measure_task2_oracle_signal.py`,
`experiments/70-t2-oracle/ORACLE_FULL.md`) midió 1214 casos, 0 saltados. El
self-test geométrico del crosshair dio error máximo **0.0000 px**. Los ratios
observado/null de vasculatura contra vasos del OCT fueron **0.7831** (r=0),
**0.9126** (r=2), **1.0311** (r=5), **1.0253** (r=10) y **1.0165** (r=20).
Las 9 proyecciones dieron z-score de NCC entre **-0.5427** y **+0.5233**. En
instrumento, **279/1214** casos (0.2298) tuvieron el keypoint dentro de la
huella; los ratios fueron **0.8750 / 0.4545 / 0.3846 / 0.5479 / 1.2110**.
Este resultado queda **parcialmente invalidado** por T2-R11b.

T2-R11b (`analysis/verify_task2_enface_convention.py`,
`experiments/70-t2-oracle/CONVENTION.md`) midió 1214 casos con radios 0.005,
0.01, 0.02 y 0.04 como fracción del lado del cuadrado unitario. En el test de
instrumento (n=469 puntos, `null_n=93800`), la convención usada hasta entonces
`identity__keep_u__keep_v` dio **1.1747 / 1.1561 / 0.7491 / 0.6995**; la
variante `transpose__flip_u__flip_v` dio **3.6364 / 3.4309 / 4.1775 /
3.1568**. En crudo a r=0.02, observado **0.0682** contra null **0.0163**.

El veredicto automático `not_convention` es un fallo de la condición: exigía
un ganador único >=1.5 y dos variantes lo superaron. Los datos sí muestran
que la convención anterior estaba mal. Matiz: el test de vasculatura no mejora
con la variante ganadora; sus ratios son **0.9444 / 0.9055 / 0.9503 /
0.9895**.

### Task 2 — T2-80, ablation por componentes

Checkpoint `task2_fixed/model_1.pth`, 248 casos de validación,
GroupKFold(5) fold 0, seed 0:

| Variante | Error medio px | Mediana px | AUC |
|---|---:|---:|---:|
| modelo tal cual | 66.25 | 41.45 | 0.0048 |
| posición GT | 22.80 | 19.67 | 0.0480 |
| rotación GT | 57.95 | 35.25 | 0.0114 |
| escala GT | 62.68 | 37.12 | 0.0187 |
| rotación + escala GT | 53.89 | 30.99 | 0.0337 |
| posición + escala GT | 18.74 | 13.63 | 0.1459 |
| posición + rotación GT | 10.78 | 9.17 | 0.2416 |
| TODO GT (control) | 3.48 | 2.77 | 0.6389 |

Al sustituir cada componente, el error cae **43.45 px (65.6%)** por posición,
**8.31 px (12.5%)** por rotación y **3.57 px (5.4%)** por escala. Los errores
medianos son: angular **5.7°**, escala **7.0%** y posición **32.9 px**.

**Consecuencia**: el techo del enfoque actual es **0.6389**, por encima del
líder del leaderboard (**0.475**). La posición explica el 66% del error; la
escala, antes señalada como posible cuello, es el componente que menos pesa
dentro de este fold.

### Task 2 — T2-80b, control OCT-shuffle

| Métrica | paired | OCT-shuffle | delta |
|---|---:|---:|---:|
| M1 top-1% | 0.003402 | 0.002457 | 0.000945 |
| M2 <=10 px | 0.068548 | 0.004032 | 0.064516 |
| M2 mediana px | 37.121 | 107.296 | -70.175 |

El modelo **sí usa contenido OCT específico del caso**. El gate de
`experiments/80-t2-diagnostics/PRE_REGISTRATION.md` exigía degradación de M1
y M2 bajo shuffle y queda superado para autorizar construir solver: M2 pierde
17x en recall y empeora 2.9x en distancia mediana.

Matiz: M1 apenas cambia (0.003402 → 0.002457). Los descriptores densos por
celda casi no llevan información específica del caso; el mapa de correlación
agregado sí. El modelo localiza la región, pero no clava el punto.

---

## Auditoría de rendimiento e I/O (2026-08-17)

Toda la sesión se atribuyó la lentitud a "MooseFS es lento". Era cierto en parte, pero **la mitad del problema era código que leía datos que nunca usaba**. Medido, no estimado:

| Hallazgo | Antes | Después |
|---|---|---|
| `find_task1_cases` abría los 61,691 JSON en un solo hilo **en cada lanzamiento** | ~15 min | **67.5 s** (paralelizado) → **~0.5 s** (cacheado en disco) |
| `Task1Dataset` cargaba el fundus (699KB, tensor float32 de 12MB) que `train_task1_unet` **nunca usa** | 38% del I/O + 100MB/batch por IPC | eliminado (`load_fundus=False`) |
| `train_task1_keypoint` cargaba los 2 B-scans + 2 máscaras que **nunca usa** | 356KB/caso muertos | eliminado (`load_bscan=False`) |
| `Task2Dataset` leía los 128 PNG del volumen (21.5MB) para promediarlos, **en cada época** | 812.7 ms/caso | **20.7 ms/caso** (en-face precomputado, 39x) |
| `train_fundus_vessel_seg` con `num_workers=0` y el volumen OCT cargado sin usarse | 97% de bytes muertos, single-thread | flags + 8 workers |
| Dataset de Task 1 leído desde MooseFS en vez de disco local | 108 ms/caso | **9.1 ms/caso** (copia local, 23GB) |
| Selección de checkpoint `if auc > best_auc` con AUC saturado en 0 | guardaba el **peor** modelo (ép5) y descartaba el mejor (ép25) | desempate por error medio |

Efecto agregado en el pod: la GPU pasó de **0-78% oscilante** a **100% sostenido**, y una época de Task 2 pasó de ~1.6 min de I/O a ~2.4 s. El precómputo del en-face redujo 26GB a 1.2GB y se verificó **idéntico bit a bit** al camino original antes de confiar en él.

**Lección transversal**: un proceso vivo con bajo %CPU sobre un filesystem de red es tan sospechoso como uno colgado, y "el filesystem es lento" es una explicación que tapa bugs de código. Cada campo que un `__getitem__` produce debe tener un consumidor verificado con grep; si no lo tiene, es I/O puro tirado a la basura multiplicado por el número de épocas.

**Cicatriz de rendimiento (2026-08-17)**: el relanzamiento tras el fix de reintentos se quedó **4.5 horas en la época 1** de T2-R2 sin terminar (CPU time 57min sobre 4.5h de reloj, ~21% de uso — I/O-bound puro). Causa: `DataLoader` sin `num_workers` (síncrono, un solo hilo leyendo de MooseFS secuencialmente, y `Task2Dataset.__getitem__` carga 128+128 slices PNG por caso). Fix: `--num-workers 8` (default nuevo) + `persistent_workers`+`prefetch_factor=4` en `train_task2_baseline.py` y `train_task1_unet.py`. Verificado en el pod: 8 procesos worker corriendo en paralelo tras el relanzamiento, época 1 de train completó en ~7 min (antes: no terminaba ni en 4.5h). **Segundo hallazgo, mismo relanzamiento**: la validación de la época 1 se quedó otros ~15 min con los workers casi sin avanzar CPU (~1% de uso) — causa: `train_task2_baseline.py` instanciaba `Task2Dataset(args.root)` con el default `include_vessel_enface=True`, cargando el volumen de segmentación completo (128 PNGs extra por caso) **aunque el modelo T2-R2 nunca usa ese campo** (el motor de vasos T2-R4 que sí lo necesitaba ya está refutado, ver arriba) — dobla el I/O por caso para nada. Fix: `include_vessel_enface=False` explícito, mismo patrón que ya usaba `train_fundus_vessel_seg.py`. Lección: cuando cambias de hipótesis (vasos descartados) revisa también los flags de carga de datos en el código que ya no depende de esa hipótesis, no solo el código que la usaba directamente.

**Tercer hallazgo, mismo relanzamiento**: aun con ambos fixes, la validación de la época 1 (fold 0: 248 casos, cada uno con 128 slices del volumen en-face) tardó **más de 1 hora sin terminar** — muy por encima de lo proporcional al train (966 casos, 5 min): 13x más lento por caso, no 4x. Verificado con `find_task2_cases`+`group_kfold_indices` en el pod (read-only, sin tocar el training) que el split real es train=966/val=248 (ratio ~3.9x), así que la desproporción es real, no un artefacto del tamaño del fold. Verificado también con inspección de `/proc/<pid>/fd` que los workers seguían leyendo archivos reales y distintos (no un loop trabado) — es latencia de MooseFS pura, agravada porque 8 workers de train (`persistent_workers=True`) siguen vivos en paralelo a los 8 de val, compitiendo por el mismo mount de red. Se descartó cachear el dataset en disco local del pod (`/dev/nvme0n1`, 3.4TB libres) porque el overlay raíz del contenedor solo expone 59GB — no cabe el dataset de Task2 (~165GB). Fix aplicado: `--val-every 5` (valida cada 5 épocas + siempre la última) en vez de cada época — reduce el I/O de validación a ~20% del total sin cambiar el entrenamiento en sí. Lección: en datasets con carga por caso muy pesada (128 slices), validar cada época es un desperdicio de tiempo, no solo de cómputo — decidir la cadencia de validación es parte del diseño del training loop, no un detalle.

---

## Hallazgos que no son corridas

Cosas medidas que cambian el diseño sin ser un experimento con score.

| Fecha | Hallazgo | Consecuencia |
|---|---|---|
| 2026-08-14 | La doc de Codabench contradice al código de ingestión en 8 puntos | `RULES_OF_ENGAGEMENT.md`. Probablemente explica varios 0.000 del leaderboard |
| 2026-08-14 | El GT de Task 2 **es** la geometría del crosshair del iOCT (coincide a 4 decimales) | El problema es geométrico, no volumétrico |
| 2026-08-14 | El crosshair **no está renderizado** en la imagen | Muere el atajo de detectarlo visualmente |
| 2026-08-15 | La matriz es una similitud reflejada de 4 DOF: det negativo 100 %, ortogonalidad 100 % sobre 1214 casos | Parametrizar 4 DOF + residuo, no 6 libres |
| 2026-08-15 | La escala **no** es constante: 160 ± 13.7 (una medición previa sobre 1 escenario daba 217 ± 5 y era engañosa) | Hay que predecirla |
| 2026-08-15 | El dataset es 8× más grande de lo anunciado: 1214 snapshots en T2, 61,691 frames en T1 | Baja el riesgo de sobreajuste |
| 2026-08-15 | Cargar los 128 PNG cuesta 46 s por 100 casos, fuera de la ventana cronometrada pero dentro del timeout global | En Final Round quedan ~5.5 s/caso reales |
| 2026-08-15 | La proyección en-face muestra vasculatura y la silueta del instrumento, alineadas con el fundus | Es la señal explotable para Task 2 |
| 2026-08-15 | Paper de 3 de los 7 organizadores existe y resuelve Task 1 exacto: arXiv 2603.25555, verificado real (no alucinado) | kp_dist 7.93px / dMAE 128.32µm son la vara de precisión alcanzable |
| 2026-08-15 | La máscara de segmentación del volumen OCT SÍ separa vasos limpio (clase 3 = ArteriesOrVeins, igual que en el fundus) | T2-R4 (puente por vasos) viable sin ambigüedad, verificado sobre el Mock Test |
| 2026-08-15 | **La distancia herramienta-tejido de Task 1 SE PUEDE MEDIR**: `dist=0.732·pixel_gap+2.678`, R²=0.9916 sobre 92,012 mediciones (61,691 frames). CANNULA activa en 100% de los frames | T1-R5: segmentación + geometría, no regresión ciega. Reduce drásticamente la varianza esperada en la componente que pesa 0.3 del score de Task 1 |
| 2026-08-15/16 | **El script de extracción de datos en el pod CORROMPÍA los datos en silencio**: extraía los 20 zips de escenario directo a `Task1/`/`Task2/` planos (sin subcarpeta por escenario). Cada zip numera sus frames desde `00000` internamente, así que Scenario_02 pisaba los JSON/PNG de Scenario_01 con el mismo `frame_id`, sin error visible (`unzip -o`). Se detectó al notar que el Task1Dataset devolvía 0 casos contra la estructura plana | Extracción parada, 46GB corruptos borrados (zips crudos intactos, cero pérdida real), script reescrito para extraer cada zip a `Task1/Scenario_XX/`, relanzado en background (`nohup`). Cicatriz: nunca confiar en que un script de extracción preserva la agrupación por escenario sin verificarlo explícitamente |
| 2026-08-16 | **Bug real en `compose_similarity(reflect=False)`**: nunca invertía el signo de `b`, así que la rama "similitud propia" no era una matriz de rotación válida (no ortogonal). Nunca se detectó porque el test que lo cubría (`test_closed_form_recovers_known_transform`) abortaba antes por falta de datos en el pod — un solo test fallando mataba el script entero sin correr los demás | Corregido en `geometry.py`. La rama `reflect=True` (la única que usa el GT real de Task 2, ya verificada al 100%) no se tocó. `test_geometry.py` reescrito para correr los 5 checks aunque alguno falle, y para reportar todos los fallos, no solo el primero. Cicatriz: un test runner que aborta en el primer fallo esconde bugs reales detrás de un fallo de datos trivial |
| 2026-08-17 | **MooseFS (fs de red del pod) da errores transitorios de I/O que se disfrazan de bugs distintos según dónde pegan**: `OSError: [Errno 6] No such device or address` al leer un PNG con PIL (mató el entrenamiento de Task 1 en época 3) y `PermissionError: [Errno 13]` al hacer `.stat()`/`.is_dir()` sobre un directorio de Task 2 (coincidió con un cuelgue real de 41 min en `find_task2_cases`, no un crash limpio) — verificado que NO es un permiso roto persistente: `find` sobre los 10 escenarios de Task2 no encuentra ningún archivo/directorio sin `r`/`x` para el owner después del `chmod -R` ya aplicado | Reintentos (3 intentos, 0.3s de espera) en `load_rgb`/`load_grayscale`/`load_label_map`/listado de PNGs de volumen (`common.py`, decorador `_retry_on_io_errors`). En `task2.py`: `find_task2_cases` captura `OSError` por caso individual y lo salta con aviso (ya no puede colgarse en un `.stat()` bloqueante de un solo caso roto); `Task2Dataset.__getitem__` captura `OSError` tras agotar reintentos y cae al siguiente índice en vez de tronar el batch completo. Relanzado T2-R2 (25 épocas) y T1-R5 (15 épocas) con el fix, script `infra/orchestrate_retry.sh` |
| 2026-08-16 | **El dataset extraído pesa ~2.5-3x lo estimado**: zips crudos 84GB (56 Task1 + 28 Task2, coincide con la estimación original) pero extraído es otra cosa — Task1 completo (10 escenarios) extrapola a ~330GB, Task2 a ~165GB, total ~495GB. El volumen de RunPod tenía 200GB asignados — la extracción murió a medio Scenario_07 con "disk full?" esperando un `y/n` que nunca iba a llegar (corría con `nohup`, sin terminal) | Volumen `<VOLUME_ID>` redimensionado de 200GB a 600GB vía API REST de RunPod (`PATCH /v1/networkvolumes/{id}`, sin necesidad de parar el pod, sin perder datos). Extracción relanzada, retoma desde Scenario_07. Cicatriz: verificar la cuota REAL del volumen (`GET /v1/networkvolumes`), no solo `df` — un mount de red (MooseFS en este caso) puede reportar el espacio libre del POOL COMPARTIDO del datacenter, no la cuota que de verdad tiene asignada este volumen |
| 2026-08-18/19 | Escala de Task 2: train (1214 casos) **[126.66, 188.09]**; Mock Test (5 casos) **[213.81, 226.16]**; solapamiento ninguno. Dentro de cada escenario, sd 1.8-4.2 px; medias entre escenarios 129.00 a 177.14. `properties.json` es idéntico en los 10 escenarios de train y en el Mock | Riesgo abierto contra el test oculto, pero no cuello de validación: T2-80 atribuye a escala **3.57 px (5.4%)**, el menor componente del fold |
| 2026-08-18/19 | Aunque `properties.json` declara `BscanScanningPattern: "Cross"`, el volumen es un ráster de 128 B-scans paralelos: correlación consecutiva media **0.671**, mínimo **0.617**, sin discontinuidad en el índice 64; correlación entre medias de slices 0-63 y 64-127 = **0.857** | La proyección en-face es geométricamente válida; no reabrir la sospecha de patrón Cross |
| 2026-08-19 | **El backbone no es el cuello de Task 1.** Tres arquitecturas, mismo split (n=14399): CNN desde cero `val_keypoint_auc=0.8539` (mejor en época 5), ResNet-18+FPN pre-entrenado `0.8534` (mejor en época **2**), DINOv2 `0.8005`. Las dos primeras difieren en 0.0005. Ambas saturan antes de la época 5 y las 15+ épocas restantes solo bajan `train_loss` sin mover el AUC de validación | T1-R4 (HRNet) y T1-R8 (ensamble) pierden justificación: atacan un eje plano medido tres veces. El techo ~0.853 es del planteamiento, no de la capacidad. Siguiente paso obligado: ablation del error residual de Task 1, análoga a T2-80 |
| 2026-08-19 | **T2-82 (descriptores comunes CNN) RECHAZADO.** 10 épocas: `train_loss` 1.8e-2 → 3.7e-7 (colapso a solución trivial), `m1_top1pct` 0.0618 → 0.0757 (ep6) → 0.0573, `m2_median_px` 178 → 156 → 187 px. **`shuffle_drop_points` oscila entre -0.567 y +0.151, ruido alrededor de cero**: barajar el OCT entre casos no degrada nada | El modelo no usa el OCT. Pierde el control OCT-shuffle de T2-80b que el baseline de T2-R2 sí superaba. La ruta "espacio común aprendido por apariencia" queda cerrada — no por hiperparámetros, sino por ausencia de señal |
| 2026-08-19 | **T2-83, oráculo de convención en-face sobre 966 casos de train (`Mock used=false`): `gate_pass=true` para `transpose__flip_u__flip_v`, pero el gate lo aprueba SOLO el instrumento (312 puntos, ratio 3.44/3.40/3.60/3.01 por radio). La vasculatura (906 puntos) da 32 ratios en [0.81, 1.04] — plana en las 8 orientaciones** | La orientación en-face es correcta y no hay que reabrirla. Pero la señal densa que un modelo de apariencia aprendería no existe en ninguna orientación: esto **predice** el fracaso de T2-82 y es evidencia independiente del oráculo de apariencia ya negativo sobre 1214 casos |
| 2026-08-19 | Matiz sobre T2-83 que no conviene olvidar: la evidencia separa con fuerza el **par de flips**, pero transpose vs identity con margen fino — a r=0.005 `identity__flip_u__flip_v` da 4.41 contra 3.44 de la ganadora. Lo que decide es la estabilidad a través de radios (transpose 3.0-3.6 en los cuatro; identity decae 4.41 → 1.78). n=312 puntos ≈ 24 aciertos a r=0.04 | No tratar la convención como verificada al 100%: está bien elegida, pero el margen contra `identity__flip_u__flip_v` descansa en pocos aciertos. Si un peldaño futuro depende críticamente de la transposición, re-medirlo con más puntos |
| 2026-08-19 | **Los organizadores corrigieron la resolución de profundidad y publicaron el scoring nuevo.** Descargado de `SynthesEyes-GmbH/fido-2026` y diffeado contra `vendor/` congelado: **solo cambia `scoring_keypoints.py`**, en dos constantes — `GROUND_TRUTH_DISTANCE_SCALE` 10 → `(4.0/512)*1000` = **7.8125**, y `MAX_THRESHOLD_DIST` 10 → **20**. `scoring_registration.py`, los dos `ingestion_*` y `run_local.py` **no cambian ni un byte** | **La corrección NO toca Task 2**, solo la componente de distancia de Task 1 (peso 0.3). `vendor/` re-sincronizado al upstream nuevo y verificado idéntico; `GROUND_TRUTH_DISTANCE_SCALE` corregido en `src/fido/data/task1.py` y en los 3 scripts de `analysis/`. 148 tests pasan |
| 2026-08-19 | **La distancia se estaba prediciendo con un sesgo sistemático del 28% por lo bajo.** El ajuste `dist = 0.7320·pixel_gap + 2.6779` (R²=0.9916) se ajustó contra `GT[2]/10`; el target real es `GT[2]/7.8125`, es decir **1.28× mayor**. Con el ejemplo del propio scoring (`stored 613.7916`): predecíamos 61.38 px donde la verdad son 78.69 px — **17.3 px de error sistemático contra un umbral de 10 px** | **Explica el `distance_auc=0.0900` de Codabench sin necesidad de culpar al modelo**: no fallaba la segmentación, fallaba la escala. Corrección exacta y sin reentrenar: multiplicar ambas constantes por 1.28 → `0.9370·pixel_gap + 3.4277` |
| 2026-08-19 | **Confirmación independiente de que la corrección es la correcta**: la pendiente ajustada pasa de **0.732 → 0.937**. Si `GT/7.8125` es de verdad la distancia en píxeles del B-scan, la pendiente contra `pixel_gap` debe ser ≈1.0. Con el factor viejo estaba 27% desviada; con el nuevo, a 6.3% | No es solo que el correo lo diga: nuestros propios 92,012 datos de T1-R2 lo corroboran. El 6.3% residual queda como pregunta abierta (clase de segmentación, punto de referencia exacto) |
| 2026-08-19 | **Aritmética del score verificada contra `r01`**: `0.7·0.7309 + 0.3·0.0900 = 0.5386`, exacto al dígito publicado. Estado del leaderboard: `r01=0.5386`, `r03=0.569`, `r04=0.561` (Task 1) y **`0.00` en Task 2**, líder **0.619** | Con `distance_auc=0.0900` estamos cobrando 0.027 de los 0.30 que reparte la distancia. **Cerrar el hueco al líder (0.050) requiere subir `distance_auc` de 0.09 a 0.257** — y eso antes de contar que el umbral subió de 10 a 20 px, que sube el AUC gratis. El margen del proyecto está aquí, no en Task 2 |
| 2026-08-19 | Extras del anuncio: **+10 submissions** para todos y **fase Competition extendida 2 días** (cierra **22 ago**, no 20) | Deja de ser cierre inmediato: hay margen para subir una submission con la distancia corregida y medirla de verdad |
| 2026-08-19 | **Control aislado del reescalado de distancia** (Mock Test, 5 casos, `keypoint_auc=0.854545` idéntico en las tres corridas, así que la distancia es lo único que se mueve): `r04` con scorer viejo `distance_auc=0.0909`; **`r04` con scorer nuevo `0.0000`**; **`r05` reescalado con scorer nuevo `0.1333`**. Scores Task 1: 0.6255 / 0.5982 / **0.6382** | **La subida del umbral (10→20 px) no aporta nada por sí sola**: bajo el scorer corregido la predicción vieja cae a cero exacto, o sea su error supera los 20 px en todos los casos. Toda la ganancia viene del reescalado. Eso convierte la corrección en **obligatoria, no opcional** — subir `r04` sin tocar tras el anuncio habría *perdido* los 0.027 que la distancia aportaba. ⚠️ 5 casos confirman la dirección, no la magnitud; el Mock no selecciona modelos (`CONSTITUTION.md` §3) |
| 2026-08-19 | **El `0.00` de Task 2 en `r04` NO es una trampa de scoring silencioso** (hipótesis mía, refutada). `submissions/r04-interim-joint/local_score_task2.json` ya registraba `final_score: 0.0` sobre el Mock con 0 timeouts **antes de subir**, con el scorer vendorizado sin modificar, y el propio README de la submission lo documentaba. Contrato verificado entero: firma, `model_1.pth`, matriz `(3,3)` con fila `[0,0,1]`, `oct_volume=None`, zip plano | El cero es honesto: el modelo de Task 2 está realmente en cero. Mecanismo: `auc_from_errors` da exactamente 0 si ningún caso baja del umbral, consistente con el techo medido del checkpoint (0.0048–0.0092) más el hueco de escala sin solape. No hay bug de formato que arreglar |
| 2026-08-19 | Verificado que **DINOv2 estaba `frozen` por defecto** (`src/fido/models/task2_dinov2.py:19,31-33`), así que el `0.8005` descarta "DINOv2 congelado", no DINOv2. Pero LP→FT no ataca el cuello: **T2-82 ya entrenó una CNN completamente descongelada** y colapsó sin usar el OCT, y **T2-83 midió ausencia de correlación a nivel de oráculo, sin red de por medio** | Congelar/descongelar es un eje de capacidad; lo medido es ausencia de señal. `experiments/85-t2-dino-lpft/` está pre-registrado pero nunca se ejecutó, y sigue sin justificación fuerte |
| 2026-08-19 | 🔴 **EL LEADERBOARD REFUTA NUESTRO VEREDICTO DE TASK 2.** Consultado el endpoint público `https://www.codabench.org/api/phases/27554/get_leaderboard/`: **siete equipos puntúan por encima de cero en Task 2** (task_id 33629) — cnielsen **0.475**, Alex 0.376, oli4more 0.28, kwonmj 0.279, dancies 0.244, tianmaxingkong 0.145, akkkkk 0.111 | **Task 2 ES RESOLUBLE**, y a 0.475 sobre un techo teórico de 0.6389 con matriz GT perfecta. Los cinco cierres internos por "ausencia de señal" (apariencia, vasos, DPCN, instrumento, priors) son **incompatibles con este hecho empírico**. En algún punto de nuestra cadena hay un error: o los oráculos miden lo que no importa, o un fallo de convención/geometría aguas arriba los contamina a todos. **Se reabre Task 2 entera** |
| 2026-08-19 | **No aparecemos en el leaderboard de la fase Competition.** Las 12 entradas son de otros equipos y ninguna corresponde a nuestras submissions; en particular no existe ningún 0.569 | Puede explicar el "Submitting" trabado. Pendiente verificar si nuestras submissions llegaron a registrarse en la fase o se quedaron sin publicar al leaderboard |
| 2026-08-19 | Revisión adversarial: **nadie verificó que Codabench redesplegara el scorer corregido** — solo se verificó que cambió el repo de GitHub. Además `RULES_OF_ENGAGEMENT.md`, que `CLAUDE.md` manda leer primero, sigue documentando las constantes viejas | Toda comparación de scores entre submissions anteriores y posteriores al 19-ago está sin base hasta confirmar qué scorer corrió. Corregir `RULES_OF_ENGAGEMENT.md` es prioritario |
| 2026-08-19 | Revisión adversarial: la comparación "r05 vs r03" estaba **mal planteada**. `r05` no parte de `r03` sino de `r04-interim-joint` (hash de `model_0.pth` distinto, verificado), cuyo score real es **0.561**. La comparación causal correcta es r05 (0.569) vs r04 (0.561) = **+0.008** | No fue "no cambió nada": fue una mejora pequeña. Sigue por debajo de lo esperado, consistente con que el fallback de `r05` seguía sin reescalar (lo corrige `r06`) |
| 2026-08-19 | Revisión adversarial: medida en vivo con el checkpoint de producción sobre B-scans reales, la fracción de casos que **cae al fallback de distancia** es **36.7% en Scenario_01 y 7.5% en Scenario_02** (n=120 c/u, 2 de 10 escenarios) | Entre el 15% y el 45% de los casos reales **nunca pasan por el ajuste A·gap+B**. Confirma en dirección la sospecha: el problema de la distancia no es solo la fórmula, es que el medidor no mide. Refuerza la prioridad de la cabeza DFL, que predice siempre |
| 2026-08-19 | Revisión adversarial: **`keypoint_auc ≈ 0.774` no tiene fuente verificable** — es una inferencia aritmética asumiendo `distance_auc=0.09`, no un dato medido, y se venía usando como si lo fuera | No apoyar decisiones en ese número hasta obtener el desglose real de Codabench |
