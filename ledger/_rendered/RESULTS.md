# RESULTS — qué funcionó, qué no, y con qué evidencia

> Un renglón por corrida con número. Los fracasos cuentan igual que los éxitos (`CONSTITUTION.md` §5).
>
> **Archivo generado.** La fuente de verdad es `ledger/`; este `.md` se
> reescribe con `python -m fido.ledger render`. Editarlo a mano no sirve
> de nada: el siguiente render se lo lleva.

*Generado el 2026-08-17T17:44:08Z.*

## Corridas

| peldaño | corrida | estado | métricas | ép. | n val | fold | commit |
|---|---|---|---|---|---|---|---|
| `10-t1-distancia-unet` | `run-2026-08-17-p10-t1-distancia` | 💥 murio | `val_distance_auc` 0.5681 · `val_distance_mae_px` 7.16 | 2 | 12566 | 0/5 | sin git |
| `30-t2-baseline-roto` | `run-2026-08-17-p30-t2-baseline-roto` | ✅ completada | `val_corner_auc` 0.00 · `val_mean_error_px` 98.74 | 25 | 248 | 0/5 | sin git |
| `40-t2-decodificador-corregido` | `run-2026-08-17-p40-t2-decodificador` | ✅ completada | `val_corner_auc` 0.0048 · `val_mean_error_px` 66.25 | 60 | 248 | 0/5 | sin git |
| `50-t1-keypoint` | `run-2026-08-17-p50-t1-keypoint` | 🔄 en_curso | `val_keypoint_auc` 0.8534 · `val_mean_error_px` 1.04 | 4 | 14399 | 0/5 | sin git |

## Cómo leer cada número

Los avisos de abajo no son opcionales: un número con un defecto abierto
encima no significa lo que parece.

### `run-2026-08-17-p10-t1-distancia` — T1 distancia: UNet cánula+ILM + medición geométrica

- **Peldaño**: `10-t1-distancia-unet` · **estado**: murio
- **Métricas**: `val_distance_auc` 0.5681 · `val_distance_mae_px` 7.16
- **Comando**: `python -m fido.train.train_task1_unet --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json --class-weights 0.0341661,0.4904578,2.4753761 --epochs 15 --batch-size 8 --lr 1e-3 --n-folds 5 --fold 0 --val-every 3 --out checkpoints/task1_real`
- **Commit**: sin git · **semilla**: 0 · **host**: `<POD_ID>`
- **Cuándo**: desde 2026-08-16T00:00:00Z
- **Checkpoint**: `/workspace/checkpoints_backup/model_0_auc0.5681_ep2.pth`
- 🟠 **ABIERTO** (`def-t1-distance-auc-inflado`): El `val_distance_auc=0.5681` está inflado: `evaluate()` descarta casos que el scorer penaliza
- 🟡 **ABIERTO** (`def-t1-distancia-sync-gpu`): El bucle de distancia hace ~96 sincronizaciones GPU↔CPU por batch
- 🟠 **ABIERTO** (`def-getitem-fallback-cruza-split`): El fallback de `__getitem__` puede inyectar muestras de train en validación (y colgarse días)

12.8x el piso trivial (0.0445), muy por encima de la expectativa pre-registrada (>0.15). Murió en la época 3 por `OSError: [Errno 6]` de MooseFS, no por un bug de código: el número es de la época 2, no de una corrida completa de 15. **El AUC real de Codabench será menor**: ver el defecto `def-t1-distance-auc-inflado`.

### `run-2026-08-17-p30-t2-baseline-roto` — T2 baseline: heatmap por correlación + regresión (θ,s) — versión con techo estructural

- **Peldaño**: `30-t2-baseline-roto` · **estado**: completada
- **Métricas**: `val_corner_auc` 0.00 · `val_mean_error_px` 98.74
- **Comando**: `—`
- **Commit**: sin git · **semilla**: 0 · **host**: `<POD_ID>`
- **Cuándo**: desde 2026-08-17T00:00:00Z
- **Checkpoint**: `/workspace/checkpoints_backup/model_1_ep5_err203px.pth`
- 🔴 **cerrado** (`def-t2-centro-vs-esquina`): El pico de correlación marca el CENTRO de la plantilla, pero se asignaba a (tx,ty), que es la ESQUINA
- 🟠 **cerrado** (`def-t2-sesgo-media-celda`): Sesgo de media celda por kernel de correlación con lados PARES
- 🟠 **cerrado** (`def-t2-plantilla-4a1`): Plantilla 4:1 comparada contra una huella cuadrada
- 🟠 **cerrado** (`def-t2-escala-sin-normalizar`): El término de escala sin normalizar se comía el 97% de la pérdida
- 🟠 **cerrado** (`def-checkpoint-guarda-el-peor`): La selección de checkpoint guardaba el PEOR modelo cuando el AUC empataba

25 épocas completas, 966 train / 248 val. AUC exactamente 0.0000 en las 5 validaciones. **Esta corrida no podía puntuar aunque entrenara perfecto**: con localización perfecta sacada del GT, el decodificador viejo daba AUC 0.0000 y el corregido 0.7848. El error de 98.74 px coincide, dentro del ruido, con los 113.1 px de offset centro↔esquina medidos sobre los 1214 casos. El checkpoint guardado es el de la época 5 (el peor), por `def-checkpoint-guarda-el-peor`. Comando exacto no recuperable: la corrida es anterior a este ledger.

### `run-2026-08-17-p40-t2-decodificador` — T2: mismo modelo con los 3 bugs estructurales corregidos, 60 épocas

- **Peldaño**: `40-t2-decodificador-corregido` · **estado**: completada
- **Métricas**: `val_corner_auc` 0.0048 · `val_mean_error_px` 66.25
- **Comando**: `python -m fido.train.train_task2_baseline --root /workspace/data/Task2 --enface-cache /root/data_cache/Task2_enface --epochs 60 --batch-size 8 --lr 3e-4 --n-folds 5 --fold 0 --val-every 5 --out checkpoints/task2_fixed`
- **Commit**: sin git · **semilla**: 0 · **host**: `<POD_ID>`
- **Cuándo**: desde 2026-08-17T00:00:00Z
- **Checkpoint**: `/workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth`
- 🟠 **ABIERTO** (`def-getitem-fallback-cruza-split`): El fallback de `__getitem__` puede inyectar muestras de train en validación (y colgarse días)

Primer AUC no-cero del proyecto en Task 2. Mismo split y mismo fold que `run-2026-08-17-p30-t2-baseline-roto`, que es el control. Con entrada perfecta el decodificador nuevo da 0.7848 (el viejo, 0.0000): el suelo de cristal desapareció. **0.0048 no es competitivo** — el líder va en 0.475. El error se aplana ruidoso entre 57 y 72 px. Siguiente límite ya medido: la rotación se predice desde features promediados globalmente, y el promedio destruye la orientación.

### `run-2026-08-17-p50-t1-keypoint` — T1 keypoint: CNN + heatmap sub-píxel, primer entrenamiento real

- **Peldaño**: `50-t1-keypoint` · **estado**: en_curso
- **Métricas**: `val_keypoint_auc` 0.8534 · `val_mean_error_px` 1.04
- **Comando**: `python -m fido.train.train_task1_keypoint --root /workspace/data/Task1 --epochs 20 --batch-size 8 --lr 1e-4 --n-folds 5 --fold 0 --val-every 4 --case-cache /workspace/cache/task1_cases.json --out /workspace/FIDO_CHALLENGE/checkpoints/task1_keypoint`
- **Commit**: sin git · **semilla**: 0 · **host**: `<POD_ID>`
- **Cuándo**: desde 2026-08-17T06:00:00Z
- **Checkpoint**: `/workspace/FIDO_CHALLENGE/checkpoints/task1_keypoint/keypoint_only.pth`
- 🟠 **ABIERTO** (`def-getitem-fallback-cruza-split`): El fallback de `__getitem__` puede inyectar muestras de train en validación (y colgarse días)

**Corrida viva al sembrar este ledger** (iba por la época 7 de 20; las métricas son del último corte de validación, época 4). El keypoint pesa 0.7 del score de Task 1, así que este es el número más valioso del proyecto hasta ahora. Actualizar el registro cuando termine: `estado`, `epoca` y `metricas`. Lanzada por `infra/orchestrate_task1.sh`, log en `/workspace/t1_kp.log`.

## Cotas y relaciones medidas

Medidos, no estimados. Contexto para juzgar cualquier resultado futuro.

| cota o relación | valor | de dónde sale |
|---|---|---|
| Piso trivial de `distance_auc` en Task 1 | **0.0445** | `analysis/explore_task1_gt.py — mejor constante posible, 159.2 px` |
| Fórmula del score oficial de Task 1 | **final_score = 0.7·keypoint_auc + 0.3·distance_auc** | `vendor/fido/Codabench Bundle/scoring_program/scoring_keypoints.py:12-13,269` |
| Relación distancia↔píxeles en Task 1 | **distancia = 0.7320·pixel_gap + 2.6779 (R² = 0.9916)** | `analysis/verify_task1_distance_geometry.py — 92,012 mediciones sobre 61,691 frames` |
| Offset centro↔esquina en Task 2 | **113.1 ± 9.9 px (teórico `s·√2/2` = 112.9 con s=159.6)** | `medición sobre los 1214 casos de entrenamiento` |

## Contra qué se compite

- **Leaderboard de la fase Competition**: Task 1 líder 0.619 · Task 2 líder 0.475 *(medido el 2026-08-14)*
