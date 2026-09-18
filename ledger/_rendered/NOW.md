# NOW — dónde estamos ahora mismo

> Estado vivo del proyecto, derivado del ledger.
>
> **Archivo generado.** La fuente de verdad es `ledger/`; este `.md` se
> reescribe con `python -m fido.ledger render`. Editarlo a mano no sirve
> de nada: el siguiente render se lo lleva.

*Generado el 2026-08-17T17:44:08Z.*

## Corriendo ahora

| peldaño | corrida | script | métricas al último corte |
|---|---|---|---|
| `50-t1-keypoint` | `run-2026-08-17-p50-t1-keypoint` | `fido.train.train_task1_keypoint` | `val_keypoint_auc` 0.8534 · `val_mean_error_px` 1.04 (ép. 4/20) |

## Qué peso usar hoy

| task | componente | checkpoint | métrica | ¿sigue vivo? |
|---|---|---|---|---|
| 1 | distancia | `/workspace/checkpoints_backup/model_0_auc0.5681_ep2.pth` | `val_distance_auc` 0.5681 · `val_distance_mae_px` 7.16 ⚠️ 3 defecto(s) abierto(s) | desconocido (sin verificar nunca) |
| 1 | keypoint | `/workspace/FIDO_CHALLENGE/checkpoints/task1_keypoint/keypoint_only.pth` | `val_keypoint_auc` 0.8534 · `val_mean_error_px` 1.04 ⚠️ 1 defecto(s) abierto(s) | desconocido (sin verificar nunca) |
| 2 | registracion | `/workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth` | `val_corner_auc` 0.0048 · `val_mean_error_px` 66.25 ⚠️ 1 defecto(s) abierto(s) | desconocido (sin verificar nunca) |

Detalle completo con `python -m fido.ledger best --task N`.

## Defectos abiertos (5)

| | severidad | defecto | dónde |
|---|---|---|---|
| 🔴 | critica | `inference.py` real no existe y el preproceso de la ingestión no coincide con el del dataset | `submissions/ (no escrito) · vendor/fido ingestión` |
| 🟠 | alta | La escala del GT en train y en el Mock Test son rangos disjuntos | `datos (Task 2)` |
| 🟠 | alta | El fallback de `__getitem__` puede inyectar muestras de train en validación (y colgarse días) | `src/fido/data/task1.py y task2.py :: Dataset.__getitem__` |
| 🟠 | alta | El `val_distance_auc=0.5681` está inflado: `evaluate()` descarta casos que el scorer penaliza | `src/fido/train/train_task1_unet.py :: evaluate()` |
| 🟡 | media | El bucle de distancia hace ~96 sincronizaciones GPU↔CPU por batch | `src/fido/train/train_task1_unet.py :: bucle de distancia` |

## Alertas del ledger

**Errores**

- `defect/def-inference-no-existe` — defecto CRÍTICO abierto: `inference.py` real no existe y el preproceso de la ingestión no coincide con el del dataset

**Avisos**

- **8 registros** — efímero y nunca verificado (existe: desconocido): `asset/cache-task1-cases-json`, `asset/cache-task1-cases-local-json`, `asset/cache-task1-local`, `asset/cache-task2-enface`, `asset/ck-t1-distancia-ep2`, `asset/ck-t1-keypoint`, `asset/ck-t2-decodificador-ep60`, `asset/ck-t2-ep5-respaldo`
- `decision/dec-task2-primero` — decisión marcada para revisión: Task 2 antes que Task 1

## Por revisar (podredumbre)

| registro | edad | revisar si… |
|---|---|---|
| `fact/f-leaderboard-competition` — Leaderboard de la fase Competition | 3 d (vida útil 3) | pasa más de un par de días, o antes de decidir a qué task dedicar GPU |

## Riesgo de pérdida

8 de 8 activos viven en almacenamiento efímero (el pod). Si el pod muere, se pierden.
