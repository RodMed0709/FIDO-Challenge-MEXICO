# FIDO Task 1 Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mejorar el score oficial de Task 1 mediante medición limpia, una CNN de mayor resolución, DINOv2 LP-FT+LLRD y una cabeza de distancia evaluada sobre todos los casos.

**Architecture:** Keypoint y distancia se desarrollan como ramas independientes y se combinan solo con predicciones out-of-fold compatibles. La CNN histórica permanece como control; DINOv2 se evalúa primero frozen y luego con full fine-tuning LLRD sin cambiar simultáneamente decoder, resolución o split.

**Tech Stack:** Python, PyTorch >=2.7/cu128, timm, NumPy, pytest, GroupKFold, harness oficial vendorizado.

**Spec:** `docs/superpowers/specs/2026-08-18-fido-task1-improvement-design.md`

## Global Constraints

- No modificar `vendor/fido/`.
- Una hipótesis científica por peldaño; pre-registro antes de cada corrida.
- GroupKFold por escenario; Mock Test solo después de congelar el candidato.
- Semilla 0 para screening; comando, commit, log y predicciones por caso obligatorios.
- El gate primario es AUC oficial en umbrales enteros 0..10.
- El artefacto final usa `model_0.pth`, firma de cuatro argumentos y soporta `oct_volume=None`.
- Benchmark final `<=540 s/100` casos dentro del contenedor.
- No commitear directamente en `master`; si la rama sigue siendo `master`, dejar cambios en working tree.

---

### Task 1: Sanear split y evaluación all-case

**Files:**
- Modify: `src/fido/data/task1.py`
- Modify: `src/fido/train/train_task1_keypoint.py`
- Modify: `src/fido/train/train_task1_unet.py`
- Create: `src/fido/eval_task1.py`
- Create: `src/fido/tests/test_task1_evaluation.py`
- Create: `experiments/80-t1-clean-baselines/PRE_REGISTRATION.md`

**Interfaces:**
- Produces: `evaluate_keypoints(pred_xy, gt_xy, scenarios) -> dict`
- Produces: `evaluate_distances(pred, gt, scenarios, measured_mask) -> dict`
- Produces: CSV/JSONL con una fila por caso y métricas pooled/per-scenario.

- [ ] **Step 1: Pre-registrar que el fallback recursivo puede cruzar el Subset**

Escribir el control CNN y DINO frozen, seed 0, fold 0, sin cambios de modelo. Declarar que `0.8537/0.8005` son históricos, no baselines limpios.

- [ ] **Step 2: Escribir tests de denominador y disjunción**

```python
def test_distance_auc_keeps_unmeasurable_cases():
    out = evaluate_distances(
        pred=np.array([10.0, 159.2]), gt=np.array([10.0, 300.0]),
        scenarios=np.array(["S1", "S2"]), measured_mask=np.array([True, False]))
    assert out["n"] == 2
    assert out["coverage"] == 0.5

def test_group_split_has_no_scenario_overlap():
    train, val = first_task1_split(fake_cases_for_ten_scenarios())
    assert set(train.scenario).isdisjoint(set(val.scenario))
```

- [ ] **Step 3: Ejecutar tests y confirmar el fallo**

Run: `PYTHONPATH=src python -m pytest -q src/fido/tests/test_task1_evaluation.py`  
Expected: FAIL porque `eval_task1.py` y las funciones aún no existen.

- [ ] **Step 4: Implementar evaluación común y eliminar sustitución silenciosa**

El dataset reintenta el mismo caso un número acotado; si sigue fallando lanza una excepción con `scenario/frame_id`. La evaluación aplica fallback `159.2` a casos no medibles, conserva el denominador y reporta CDF@0..10.

- [ ] **Step 5: Ejecutar la suite**

Run: `PYTHONPATH=src python -m pytest -q src/fido/tests`  
Expected: PASS; el conteo por caso coincide con el tamaño exacto del fold.

- [ ] **Step 6: Reproducir controles limpios**

```bash
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json \
  --backbone cnn --epochs 20 --batch-size 8 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 --out checkpoints/task1_clean_cnn
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json \
  --backbone dinov2 --freeze-backbone --epochs 20 --batch-size 4 --lr 1e-4 \
  --n-folds 5 --fold 0 --seed 0 --out checkpoints/task1_clean_dino_frozen
```

- [ ] **Step 7: Cerrar el rung y commit atómico fuera de master**

Commit sugerido: `fix: make task1 validation all-case and split-safe`

### Task 2: Medir el techo real de distancia

**Files:**
- Modify: `analysis/measure_task1_distance_ceiling.py` only if the clean run exposes a bug
- Create: `experiments/81-t1-distance-ceiling/PRE_REGISTRATION.md`
- Create: `experiments/81-t1-distance-ceiling/CEILING.md`

**Interfaces:**
- Consumes: checkpoint `task1_dist_v2/model_0.pth` y cache de 61,691 casos.
- Produces: `oracle_distance_auc_all`, `distance_auc_all`, coverage y desglose por escenario.

- [ ] **Step 1: Pre-registrar la contradicción 308 vs 376.7 px**

No cambiar el script ni el checkpoint después de registrar la expectativa.

- [ ] **Step 2: Ejecutar el análisis sin Mock**

```bash
PYTHONPATH=src /workspace/venv/bin/python analysis/measure_task1_distance_ceiling.py \
  --root /root/data_cache/Task1 \
  --case-cache /root/data_cache/task1_cases_local.json \
  --checkpoint /workspace/FIDO_CHALLENGE/checkpoints/task1_dist_v2/model_0.pth \
  --repo /workspace/FIDO_CHALLENGE --device cpu --sample-size 400 --batch-size 2 \
  --output experiments/81-t1-distance-ceiling/CEILING.md
```

- [ ] **Step 3: Gate de decisión**

Si el techo all-case es `<0.25`, priorizar cabeza aprendida; si es `>=0.35`, conservar geometría como candidato. En ambos casos el baseline autoritativo es all-case, nunca `0.6113`.

- [ ] **Step 4: Registrar comando y salida real**

Commit sugerido: `docs: record task1 distance ceiling`

### Task 3: Unificar y probar decoders de heatmap

**Files:**
- Modify: `src/fido/heatmap_decode.py`
- Create: `src/fido/tests/test_heatmap_decode.py`
- Create: `analysis/evaluate_task1_decoders.py`

**Interfaces:**
- Produces: `decode_heatmap(heatmap, mode, temperature=1.0, window=7) -> Tensor[B,C,2]`
- Produces: `image_to_heatmap_xy` y `heatmap_to_image_xy` con convención UDP.

- [ ] **Step 1: Escribir tests de gaussiana fraccional, borde y bimodalidad**

```python
def test_local_decoder_ignores_remote_mode():
    heat = two_peak_heatmap(main_xy=(10.25, 12.5), distractor_xy=(50, 50))
    xy = decode_heatmap(heat, mode="local", window=7)
    assert torch.linalg.norm(xy[0, 0] - torch.tensor([10.25, 12.5])) < 0.25

def test_udp_round_trip():
    xy = torch.tensor([[0.0, 0.0], [1023.0, 1023.0], [221.3, 617.8]])
    assert torch.allclose(heatmap_to_image_xy(image_to_heatmap_xy(xy, 1024, 64), 1024, 64), xy)
```

- [ ] **Step 2: Ejecutar y observar fallo**

Run: `PYTHONPATH=src python -m pytest -q src/fido/tests/test_heatmap_decode.py`  
Expected: FAIL por ausencia del dispatcher/UDP.

- [ ] **Step 3: Implementar dispatcher y DARK sobre probabilidades positivas**

Conservar `soft_argmax_2d`, `local_soft_argmax_2d` y `dark_offset_correction` como primitivas compatibles.

- [ ] **Step 4: Ejecutar tests**

Run: `PYTHONPATH=src python -m pytest -q src/fido/tests/test_heatmap_decode.py`  
Expected: PASS, sin NaN en picos planos o de borde.

- [ ] **Step 5: Commit**

Commit sugerido: `feat: unify subpixel heatmap decoders`

### Task 4: Rungs offline de local soft-argmax y DARK

**Files:**
- Modify: `analysis/evaluate_task1_decoders.py`
- Create: `experiments/82-t1-local-decoder/`
- Create: `experiments/83-t1-dark-decoder/`

**Interfaces:**
- Consumes: logits guardados del control CNN limpio.
- Produces: AUC, CDF@0..10, PCK@1/3/5/10 y métricas por escenario para cada decoder.

- [ ] **Step 1: Guardar una vez logits y metadatos del fold 0**

Run: `PYTHONPATH=src python analysis/evaluate_task1_decoders.py --checkpoint checkpoints/task1_clean_cnn/keypoint_only.pth --cache-logits checkpoints/task1_clean_cnn/fold0_logits.npz --mode global`

- [ ] **Step 2: Pre-registrar y evaluar local como único cambio**

Run: `PYTHONPATH=src python analysis/evaluate_task1_decoders.py --logits checkpoints/task1_clean_cnn/fold0_logits.npz --mode local --window 7`

- [ ] **Step 3: Aplicar gate local**

GO si `delta_auc>=0.002`, suben PCK@1/3 y ningún escenario cae más de `0.01`.

- [ ] **Step 4: Pre-registrar y evaluar DARK como rung independiente**

Run: `PYTHONPATH=src python analysis/evaluate_task1_decoders.py --logits checkpoints/task1_clean_cnn/fold0_logits.npz --mode dark`

- [ ] **Step 5: Cerrar ambos rungs con resultados, incluidos NO-GO**

Commit sugerido: `docs: record task1 decoder ablations`

### Task 5: Entrenar UDP como único cambio

**Files:**
- Modify: `src/fido/train/train_task1_keypoint.py`
- Modify: `src/fido/models/task1_keypoint.py`
- Modify: `src/fido/models/task1_keypoint_dinov2.py`
- Modify: `src/fido/tests/test_task1_evaluation.py`
- Create: `experiments/84-t1-udp/`

**Interfaces:**
- Adds CLI: `--coordinate-mapping legacy|udp`.

- [ ] **Step 1: Escribir test de target/decoder emparejados**

Verificar que una gaussiana centrada en GT y decodificada por el modo ganador vuelve al píxel original con `<0.25 px`.

- [ ] **Step 2: Ejecutar test y confirmar fallo con mapping legacy**

- [ ] **Step 3: Conectar UDP sin cambiar arquitectura, sigma, loss ni decoder**

- [ ] **Step 4: Entrenar fold 0**

```bash
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json \
  --backbone cnn --coordinate-mapping udp --fold 0 --seed 0 \
  --epochs 20 --batch-size 8 --lr 1e-4 --out checkpoints/task1_cnn_udp
```

- [ ] **Step 5: Gate y commit**

GO si `delta_auc>=0.002`; commit sugerido `feat: add unbiased task1 coordinate mapping`.

### Task 6: CNN FPN de stride 4

**Files:**
- Modify: `src/fido/models/task1_keypoint.py`
- Modify: `src/fido/train/train_task1_keypoint.py`
- Create: `src/fido/tests/test_task1_keypoint_models.py`
- Create: `experiments/85-t1-cnn-fpn/`

**Interfaces:**
- Adds model variant: `Task1KeypointModel(variant="legacy"|"fpn")`.
- FPN output: `[B,1,256,256]` para input `[B,3,1024,1024]`.

- [ ] **Step 1: Escribir test de shapes y compatibilidad legacy**

```python
def test_fpn_outputs_stride_four_heatmap():
    out = Task1KeypointModel(variant="fpn")(torch.zeros(1, 3, 1024, 1024))
    assert out["heatmap_logits"].shape == (1, 1, 256, 256)
```

- [ ] **Step 2: Ejecutar test y confirmar fallo**

- [ ] **Step 3: Implementar laterales 1x1 y fusión top-down de strides 4/8/16**

No cambiar encoder base, loss, augmentations ni decoder ganador.

- [ ] **Step 4: Entrenar screening fold 0**

Run: `PYTHONPATH=src python -m fido.train.train_task1_keypoint --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json --backbone cnn --model-variant fpn --fold 0 --seed 0 --epochs 20 --batch-size 8 --lr 1e-4 --out checkpoints/task1_cnn_fpn`

- [ ] **Step 5: Gate**

GO si `delta_auc>=0.005`, suben PCK@1/3 y ningún escenario cae `>0.02`.

- [ ] **Step 6: Commit**

Commit sugerido: `feat: add stride-four task1 keypoint head`

### Task 7: Preparar DINOv2 LP-FT y optimizer LLRD

**Files:**
- Modify: `src/fido/models/task1_keypoint_dinov2.py`
- Create: `src/fido/train/dinov2_optim.py`
- Modify: `src/fido/train/train_task1_keypoint.py`
- Create: `src/fido/tests/test_dinov2_optim.py`
- Create: `experiments/86-t1-dino-linear-probe/`

**Interfaces:**
- Produces: `build_dinov2_param_groups(model, head_lr, last_block_lr, layer_decay, weight_decay) -> list[dict]`.
- Adds stages: `linear_probe` and `full_finetune`.

- [ ] **Step 1: Escribir tests de independencia, gradientes y LR monótono**

```python
def test_llrd_decreases_toward_early_blocks():
    groups = build_dinov2_param_groups(model, 1e-4, 1e-5, 0.75, 0.05)
    assert lr(groups, "blocks.11") > lr(groups, "blocks.0")
    assert no_decay(groups, "norm") and no_decay(groups, "bias")
```

- [ ] **Step 2: Ejecutar test y confirmar fallo**

- [ ] **Step 3: Implementar param groups, warmup 5%, cosine y BF16**

El optimizer debe incluir solo parámetros entrenables durante LP y todos durante FT.

- [ ] **Step 4: Ejecutar smoke forward/backward a 1022 px**

Registrar versión de timm/torch, VRAM pico y gradiente finito. Si OOM incluso con batch 1, checkpointing y acumulación, cerrar este rung y pre-registrar otro de resolución; no cambiarla silenciosamente.

- [ ] **Step 5: Entrenar linear probe limpio**

Run: `PYTHONPATH=src python -m fido.train.train_task1_keypoint --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json --backbone dinov2 --stage linear_probe --fold 0 --seed 0 --epochs 20 --batch-size 4 --lr 1e-4 --out checkpoints/task1_dino_lp`

- [ ] **Step 6: Commit**

Commit sugerido: `feat: add dinov2 transfer training stages`

### Task 8: Full fine-tuning DINOv2 con LLRD

**Files:**
- Modify: `experiments/87-t1-dino-full-ft/PRE_REGISTRATION.md`
- Create: `experiments/87-t1-dino-full-ft/RESULTS.md`

**Interfaces:**
- Consumes: mejor checkpoint LP y exactamente el mismo decoder/mapping.

- [ ] **Step 1: Pre-registrar una sola receta**

Head `1e-4`, último bloque `1e-5`, decay `0.75`, WD `0.05`, warmup 5%, cosine; no sweep inicial.

- [ ] **Step 2: Ejecutar full FT**

```bash
PYTHONPATH=src python -m fido.train.train_task1_keypoint \
  --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json \
  --backbone dinov2 --stage full_finetune --resume checkpoints/task1_dino_lp/keypoint_only.pth \
  --head-lr 1e-4 --last-block-lr 1e-5 --layer-decay 0.75 --weight-decay 0.05 \
  --fold 0 --seed 0 --epochs 20 --batch-size 4 --out checkpoints/task1_dino_full_ft
```

- [ ] **Step 3: Gate**

GO si supera LP `>=0.01` AUC o queda a `<=0.005` del mejor CNN con baja correlación de errores.

- [ ] **Step 4: Confirmar ganador en folds 1 y 2 antes de cinco folds**

- [ ] **Step 5: Registrar resultado y commit**

Commit sugerido: `docs: record task1 dinov2 full fine-tuning`

### Task 9: Coarse-to-fine del mejor keypoint individual

**Files:**
- Create: `src/fido/models/task1_refiner.py`
- Create: `src/fido/train/train_task1_refiner.py`
- Create: `src/fido/tests/test_task1_refiner.py`
- Create: `experiments/88-t1-coarse-to-fine/`

**Interfaces:**
- Produces: `crop_around_prediction(image, xy, size) -> crop, origin, valid_mask`.
- Produces: coordenada fina remapeada a 1024x1024.

- [ ] **Step 1: Testear round-trip de crops y padding de bordes**

- [ ] **Step 2: Implementar crop nativo 256x256 y refiner separado**

Entrenar con crop GT+jitter pre-registrado; inferencia usa solo predicción coarse.

- [ ] **Step 3: Entrenar y medir miss-rate**

- [ ] **Step 4: Gate**

GO si `delta_auc>=0.005`, mejora PCK@1/3, miss-rate `<0.1%` y mantiene presupuesto de tiempo.

- [ ] **Step 5: Commit**

Commit sugerido: `feat: add task1 local keypoint refinement`

### Task 10: Cabeza distribucional de distancia all-case

**Files:**
- Modify: `src/fido/models/unet_bscan_seg.py`
- Modify: `src/fido/train/train_task1_unet.py`
- Create: `src/fido/tests/test_task1_distance_head.py`
- Create: `experiments/89-t1-distance-head/`

**Interfaces:**
- Produces: `DistanceDistributionHead(features) -> logits[B,num_bins]`.
- Bins y rango se derivan solo de train fold después de Task 2.

- [ ] **Step 1: Escribir tests de expectation, rango y casos sin OCT**

```python
def test_distance_expectation_is_in_pixel_units():
    logits = one_hot_logits(bin_index=42, num_bins=512)
    assert torch.allclose(decode_distance(logits), torch.tensor([42.0]))
```

- [ ] **Step 2: Ejecutar test y confirmar fallo**

- [ ] **Step 3: Exponer bottleneck UNet y entrenar solo la cabeza**

Conservar segmentación congelada y usar evaluación all-case. No añadir modality dropout aún.

- [ ] **Step 4: Entrenar fold 0**

Run: `PYTHONPATH=src python -m fido.train.train_task1_unet --root /root/data_cache/Task1 --case-cache /root/data_cache/task1_cases_local.json --distance-mode head --freeze-segmenter --fold 0 --seed 0 --out checkpoints/task1_distance_head`

- [ ] **Step 5: Gate**

GO si `delta_distance_auc_all>=0.02`; joint FT posterior requiere otro rung y `>=0.01` adicional.

- [ ] **Step 6: Commit**

Commit sugerido: `feat: add all-case task1 distance head`

### Task 11: Selección OOF, ensemble condicionado y receta final

**Files:**
- Create: `analysis/select_task1_candidate.py`
- Create: `src/fido/tests/test_task1_selection.py`
- Create: `experiments/90-t1-oof-selection/`

**Interfaces:**
- Consumes: predicciones OOF de keypoint y distancia.
- Produces: score combinado exacto, correlación de errores y receta congelada JSON.

- [ ] **Step 1: Testear que no se mezclen folds/casos incompatibles**

- [ ] **Step 2: Confirmar ganadores individuales en cinco folds**

- [ ] **Step 3: Calcular `0.7*kp_auc+0.3*distance_auc` sobre el mismo conjunto**

- [ ] **Step 4: Probar ensemble solo si CNN y DINO tienen errores complementarios**

GO ensemble si mejora `>=0.005` AUC OOF; TTA es un rung separado.

- [ ] **Step 5: Congelar receta**

GO final si mejora r03 reconstruido `>=0.01` de score combinado y al menos 6/10 escenarios.

### Task 12: Empaquetado, contrato y perfil final

**Files:**
- Create: `submissions/r04-task1-candidate/inference.py`
- Create: `submissions/r04-task1-candidate/README.md`
- Create: `src/fido/tests/test_task1_submission_contract.py`
- Create: `analysis/profile_task1_inference.py`

**Interfaces:**
- Produces: `model_0.pth` y zip plano.

- [ ] **Step 1: Escribir test de firma, shape, tipos y `oct_volume=None`**

- [ ] **Step 2: Implementar inferencia autocontenida sin modificar r03**

- [ ] **Step 3: Ejecutar suite y perfil de 100 casos**

```bash
PYTHONPATH=src python -m pytest -q src/fido/tests
PYTHONPATH=src python analysis/profile_task1_inference.py --cases 100 --include-io
```

Expected: cero excepciones/timeouts y `<=540 s` total.

- [ ] **Step 4: Ejecutar única evaluación Mock del artefacto congelado**

Run: `python eval/run_local.py --submission submissions/r04-task1-candidate --task keypoints --save`

- [ ] **Step 5: Inspeccionar zip y hashes**

Raíz exacta: `inference.py`, `model_0.pth`, `requirements.txt` opcional sin `torch`/`numpy`.

- [ ] **Step 6: Registrar submission**

Commit sugerido fuera de master: `feat: package improved task1 submission`

## Calendario y stop condition

- CPU/L4: Tasks 1–5 y Task 2 de distancia mientras la GPU principal trabaja Task 2.
- RTX 5090: Tasks 6–10, condicionadas por gates; estimación 35–65 GPU-h antes de cinco folds.
- Congelar arquitectura no después del 30 de agosto; 31 agosto–3 septiembre quedan para folds, empaquetado y contingencia.
- Stop: no continuar una rama que falla su gate; conservar el mejor candidato reproducible y pasar a empaquetado.

