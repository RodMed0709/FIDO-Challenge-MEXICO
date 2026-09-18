# EL MAPA — qué existe y en qué estado

> Hechos medidos, activos y decisiones vigentes.
>
> **Archivo generado.** La fuente de verdad es `ledger/`; este `.md` se
> reescribe con `python -m fido.ledger render`. Editarlo a mano no sirve
> de nada: el siguiente render se lo lleva.

*Generado el 2026-08-17T17:44:08Z.*

## Entorno — medido, no recordado

| hecho | valor | medido | revisar si… |
|---|---|---|---|
| Disco local del contenedor | **59 GB libres, escritura a 4.3 GB/s** | 2026-08-17 | se recrea el pod o cambia la imagen del contenedor |
| Latencia de MooseFS (fs de red del pod) | **~27 ms por lectura aleatoria de archivo; ~14 ms secuencial** | 2026-08-17 | se recrea el pod, cambia el datacenter, o se remonta el volumen |
| RAM del pod | **187** GB | 2026-08-17 | se recrea el pod con otro tipo de instancia |

- **Disco local del contenedor**: Caben los B-scans de Task 1 (23 GB) y el en-face de Task 2 (1.2 GB). El fundus completo de Task 1 son 43 GB y NO cabe junto con lo anterior.
- **Latencia de MooseFS (fs de red del pod)**: Explica por qué todo cache local es obligatorio y no opcional.

## Los datos

| hecho | valor | medido | revisar si… |
|---|---|---|---|
| Peso de un caso de Task 1 | **fundus `microscope.png` 699 KB (1024×1024 RGB) · 2 B-scans de 176 KB (512×512 L) · 2 máscaras de 2 KB** | 2026-08-17 | cambia el formato de los datos |
| Tamaño del dataset de Task 1 | **61,691 casos en 10 escenarios** | 2026-08-17 | los organizadores publican datos nuevos |
| Peso de un caso de Task 2 | **128 PNG de volumen = 21.5 MB; su proyección en-face precomputada, 65 KB** | 2026-08-17 | cambia el formato del volumen o la resolución |
| Escala del GT: train contra Mock Test | **train 159.6 ± 13.6 px (rango 125.9–196.9); Mock Test 215.3 ± 5.6 (rango 207.1–223.6) — RANGOS DISJUNTOS** | 2026-08-17 | se compara el `properties.json` del simulador en ambos conjuntos y aparece la causa, o los organizadores aclaran la diferencia |
| Tamaño del dataset de Task 2 | **1,214 casos en 10 escenarios** | 2026-08-17 | los organizadores publican datos nuevos |

- **Tamaño del dataset de Task 1**: La documentación anuncia ~25,000. El dataset real es 2.5x más grande.
- **Peso de un caso de Task 2**: 330x menos. Por eso el cache de en-face convierte 26 GB en 1.2 GB.
- **Escala del GT: train contra Mock Test**: Es el hecho que sostiene el defecto `def-escala-train-vs-mocktest`.
- **Tamaño del dataset de Task 2**: La documentación anuncia 150. Son 8x más.

## Activos — checkpoints

| | task | componente | ruta | producido por | ¿vive? |
|---|---|---|---|---|---|
| ★ | 1 | distancia | `/workspace/checkpoints_backup/model_0_auc0.5681_ep2.pth` | `run-2026-08-17-p10-t1-distancia` (`val_distance_auc` 0.5681 · `val_distance_mae_px` 7.16) | desconocido |
| ★ | 1 | keypoint | `/workspace/FIDO_CHALLENGE/checkpoints/task1_keypoint/keypoint_only.pth` | `run-2026-08-17-p50-t1-keypoint` (`val_keypoint_auc` 0.8534 · `val_mean_error_px` 1.04) | desconocido |
| ★ | 2 | registracion | `/workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth` | `run-2026-08-17-p40-t2-decodificador` (`val_corner_auc` 0.0048 · `val_mean_error_px` 66.25) | desconocido |
|  | 2 | registracion | `/workspace/checkpoints_backup/model_1_ep5_err203px.pth` | `run-2026-08-17-p30-t2-baseline-roto` (`val_corner_auc` 0.00 · `val_mean_error_px` 98.74) | desconocido |

★ = el que se usa ahora mismo para esa task+componente.

## Activos — caches

Sin estos, un entrenamiento pasa de minutos a horas. No son opcionales.

| cache | ruta | host | ¿vive? |
|---|---|---|---|
| Lista de casos de Task 1 (MooseFS) | `/workspace/cache/task1_cases.json` | `pod:<POD_ID>` | desconocido |
| Lista de casos de Task 1 (disco local) | `/root/data_cache/task1_cases_local.json` | `pod:<POD_ID>` | desconocido |
| Dataset de Task 1 en disco local (23 GB) | `/root/data_cache/Task1` | `pod:<POD_ID>` | desconocido |
| Proyecciones en-face de Task 2 (1.2 GB) | `/root/data_cache/Task2_enface` | `pod:<POD_ID>` | desconocido |

> ⚠️ **Todo lo marcado como efímero vive solo en el pod.** Son 8 activos. Si el pod desaparece, hay que reconstruirlos o volver a entrenar.

## Decisiones

### Vigentes

**La distancia de Task 1 se MIDE por segmentación, no se regresa a ciegas** *(desde 2026-08-15)*

- Qué: Segmentar cánula e ILM en el B-scan y aplicar la relación lineal verificada.
- Por qué: R² = 0.9916 sobre 92,012 mediciones reales: la relación es casi determinista.
- Se revierte si: el `distance_auc` corregido (sin descartar casos) cae por debajo de lo que daría una regresión directa, o el 9.03% de casos fuera de rango resulta ser mayoría en el test oculto

**Dos modelos independientes en un solo zip** *(desde 2026-08-14)*

- Qué: `model_0.pth` para Task 1 y `model_1.pth` para Task 2; `inference(task_id, ...)` ramifica.
- Por qué: El formato lo permite y no hay incentivo de empaquetado para compartir backbone entre tareas heterogéneas; sí hay riesgo de interferencia.
- Se revierte si: el contrato de entrega cambia, o aparece evidencia de transferencia útil entre tasks

**No hardcodear priors de escala en Task 2** *(desde 2026-08-17)*

- Qué: Ningún valor de escala calibrado en train entra al modelo como constante.
- Por qué: Los rangos de escala de train y del Mock Test son disjuntos; un prior de train estaría sesgado si el test oculto se parece al Mock Test.
- Se revierte si: se resuelve la discrepancia train↔Mock Test y se entiende de dónde sale

**No reintentar template matching clásico sobre vasos (Task 2)** *(desde 2026-08-17)*

- Qué: El peldaño 20 queda cerrado. No se vuelve a intentar bajo ninguna variante morfológica simple.
- Por qué: Refutado con evidencia oráculo, no por un bug: con los parámetros REALES del GT (sin búsqueda) el NCC entre el patch alineado y el fundus es ≈0 en los 4 casos (-0.0072, 0.0462, -0.0116, 0.0001). El caso sintético autoconsistente pasa con 0.09 px de error, así que la cadena es matemáticamente correcta: la señal no existe.
- Se revierte si: aparece una representación de la señal de vaso que no sea una transformación morfológica simple del `enface_vessel_density` — por ejemplo probabilidades de vaso aprendidas, o combinar con la silueta del instrumento, que sigue sin explotarse

### En revisión

**Task 2 antes que Task 1** *(desde 2026-08-14)*

- Qué: Dedicar la GPU y el tiempo de diseño a Task 2 antes que a Task 1 (`CONSTITUTION.md` §14).
- Por qué: El campo estaba más débil: líder en 0.475, segundo en 0.296, cuatro equipos en 0.000. Task 1 estaba apretado (0.619 contra 0.618).
- Se revierte si: Task 1 empieza a rendir mucho más por hora invertida que Task 2 — que es exactamente lo que pasó el 2026-08-17: keypoint 0.8534 contra registración 0.0048
