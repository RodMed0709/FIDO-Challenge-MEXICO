# FIDO Task 2 Cross-Modal Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un registrador iOCT-fundus que aprenda correspondencias cross-modales, busque explícitamente la pose y supere el baseline de `corner_auc=0.0048` sin depender de priors de escenario.

**Architecture:** Dos encoders independientes producen descriptores densos comunes. Un solver PyTorch busca reflexión fija, rotación, escala y traslación, conserva top-K y ajusta una similitud reflejada mediante Umeyama/Procrustes; un refinador local es opcional y solo se habilita si el globalizador ya logra AUC no trivial.

**Tech Stack:** Python, PyTorch >=2.7/cu128, timm, NumPy, pytest, GroupKFold, `grid_sample`, FFT/correlación PyTorch y harness oficial vendorizado.

**Spec:** `docs/superpowers/specs/2026-08-18-fido-task2-cross-modal-design.md`

## Global Constraints

- No modificar `vendor/fido/`.
- Una hipótesis científica por peldaño y pre-registro antes de ejecutar.
- Selección exclusivamente con GroupKFold por escenario y estadísticas del train fold.
- Poner en cuarentena toda etiqueta/rango exacto conocido del Mock Test; Mock solo después de congelar.
- Métrica final: `corner_auc` oficial 0..10, pooled y por escenario.
- Semilla, comando, commit, log, configuración y predicciones por caso obligatorios.
- Solo PyTorch/timm mantenidos; no añadir stacks de investigación frágiles.
- Artefacto final: firma de cuatro argumentos, `model_1.pth`, matriz `float64 (3,3)`, fallback para `oct_volume=None` y `<=540 s/100`.
- No commitear directamente en `master`.

## Literatura que gobierna las decisiones

La literatura no es decorativa: cada bloque arquitectónico existe porque un
paper aporta evidencia transferible y tiene una prueba local que puede
refutarlo. La revisión completa, con tablas y PDFs leídos, está en
`literature/04_task2_cross_modal.md`.

| Decisión del plan | Evidencia primaria | Qué se toma y qué no se asume |
|---|---|---|
| Aprender un espacio común antes de registrar | [CoMIR, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html) | Se toma InfoNCE densa y equivariancia; no se trasladan sus scores BF-SHG a retina. |
| Tratar Task 2 como sub-image retrieval cross-modal | [Cross-modality sub-image retrieval using CoMIRs, Scientific Reports 2024](https://www.nature.com/articles/s41598-024-68800-1) | Se toma retrieval top-K + reranking; su top-K no sustituye el AUC FIDO. |
| Reducir el espacio global antes del matcher fino | [CARe, 2026](https://arxiv.org/abs/2512.12657) y [Two-Step Retinal Registration, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8912939/) | Se toma cascada coarse-to-fine ante gran disparidad de FOV; no se copia su señal OCTA/vasos. |
| Adaptar el matcher a ambas modalidades | [KPVSA-Net, 2022](https://arxiv.org/abs/2207.10506) y [XoFTR, 2024](https://openaccess.thecvf.com/content/CVPR2024W/IMW/html/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.html) | Se toman encoders/adaptadores específicos, dustbin y refinamiento subpíxel; no checkpoints off-the-shelf. |
| Buscar rotación/escala geométricamente | [DPCN++, 2022](https://arxiv.org/abs/2206.05707) | Se toma el desacoplamiento log-polar/phase-correlation sobre features aprendidas; nunca sobre intensidades crudas ni antes del gate de representación. |
| Conservar hipótesis múltiples y ajustar 4-DOF | [KPVSA-Net](https://arxiv.org/abs/2207.10506) y [Retinal IPA, 2024](https://arxiv.org/abs/2407.18362) | Se toman correspondencias, rechazo de outliers y ajuste robusto; FIDO restringe el solver a similitud reflejada. |
| Refinar solo dentro de una cuenca útil | [Iterative Deep Homography Estimation, 2022](https://arxiv.org/abs/2203.15982) | Se toma el updater iterativo local; queda prohibido usarlo como globalizador. |
| No priorizar síntesis generativa | [Comparative study of I2I for registration, 2022](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276196) | La traducción visual no predice registro cuando las modalidades observan propiedades distintas; cualquier síntesis futura debe ganar el mismo gate geométrico. |

Trazabilidad directa:

- Tasks 3–6 implementan/adversan CoMIR, KPVSA y XoFTR.
- Tasks 7–8 implementan/adversan DPCN++ después del gate común.
- Task 9 implementa top-K y ajuste restringido inspirado en KPVSA/Retinal IPA.
- Task 10 implementa la idea local de IHN únicamente si el globalizador pasa.
- Tasks 1–2 son evidencia local previa: si contradicen un paper, manda FIDO.

---

### Task 1: Congelar diagnósticos y baseline autoritativo

**Files:**
- Modify: `analysis/diagnose_task2_error.py`
- Modify: `analysis/measure_task2_correspondence.py`
- Create: `src/fido/eval_task2.py`
- Create: `src/fido/tests/test_task2_evaluation.py`
- Create: `experiments/80-t2-diagnostics/PRE_REGISTRATION.md`
- Create: `experiments/80-t2-diagnostics/RESULTS.md`

**Interfaces:**
- Produces: `evaluate_task2_cases(pred_matrix, gt_matrix, scenarios) -> dict`.
- Produces: sustitución GT por posición, rotación y escala con AUC 0..10.
- Produces: M1–M3 y control OCT-shuffle sobre exactamente el mismo fold.

- [ ] **Step 1: Pre-registrar tres sustituciones independientes**

No usar split aleatorio como evidencia. La escala es sospechosa hasta que la sustitución GT y un rung de augmentación la aíslen.

- [ ] **Step 2: Escribir test contra `corner_auc` oficial**

```python
def test_task2_case_evaluator_matches_geometry_auc():
    out = evaluate_task2_cases(pred, gt, np.array(["S1", "S2"]))
    expected = corner_auc(corner_error(pred, gt))
    assert out["auc"] == expected
```

- [ ] **Step 3: Ejecutar test y confirmar fallo**

Run: `PYTHONPATH=src python -m pytest -q src/fido/tests/test_task2_evaluation.py`

- [ ] **Step 4: Implementar evaluator y JSONL por caso**

- [ ] **Step 5: Ejecutar diagnóstico en fold 0**

```bash
PYTHONPATH=src python analysis/diagnose_task2_error.py \
  --root /workspace/data/Task2 --enface-cache /root/data_cache/Task2_enface \
  --checkpoint /workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth \
  --fold 0 --device cuda
PYTHONPATH=src python analysis/measure_task2_correspondence.py \
  --root /workspace/data/Task2 --checkpoint /workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth \
  --n-folds 5 --fold 0 --k 64 --device cuda --oct-shuffle
```

- [ ] **Step 6: Cerrar diagnóstico**

El resultado decide prioridad; no autoriza por sí mismo una submission. Commit sugerido: `test: establish task2 diagnostic baselines`.

### Task 2: Augmentación geométrica exacta derivada de train

**Files:**
- Create: `src/fido/data/task2_transforms.py`
- Modify: `src/fido/data/task2.py`
- Modify: `src/fido/train/train_task2_baseline.py`
- Create: `src/fido/tests/test_task2_transforms.py`
- Create: `analysis/evaluate_task2_scale_holdout.py`
- Create: `experiments/81-t2-scale-augmentation/`

**Interfaces:**
- Produces: `apply_fundus_homography(fundus, gt_matrix, image_h) -> tuple[Tensor,Tensor]`.
- Contract: `M_aug = H @ M`; rango de augmentación calculado solo desde train fold.

- [ ] **Step 1: Testear composición, determinante y esquinas**

```python
def test_fundus_transform_composes_ground_truth():
    image2, m2 = apply_fundus_homography(image, m, h)
    assert torch.allclose(m2, h @ m, atol=1e-5)
    assert corner_error(m2, h @ m).item() < 1e-3
    assert torch.det(m2[:2, :2]) < 0
```

- [ ] **Step 2: Ejecutar test y confirmar fallo**

- [ ] **Step 3: Implementar crop/resize/rotate con valid mask**

Las augmentaciones fotométricas son independientes por modalidad y los artefactos de padding nunca cuentan como correspondencia.

- [ ] **Step 4: Definir holdout sintético sin Mock**

Derivar cuantiles y extrapolación log-scale desde los diez escenarios de train. Reservar una banda pre-registrada fuera del soporte del train fold; no copiar `213–226`.

- [ ] **Step 5: Entrenar baseline con augmentación como único cambio**

Run: `PYTHONPATH=src python -m fido.train.train_task2_baseline --root /workspace/data/Task2 --enface-cache /root/data_cache/Task2_enface --fold 0 --seed 0 --scale-augment --out checkpoints/task2_scale_aug`

- [ ] **Step 6: Gate**

GO si sube `corner_auc` y baja al menos 20% el error de escala en holdout sintético sin perjudicar más de `0.01` AUC por escenario.

- [ ] **Step 7: Commit**

Commit sugerido: `feat: add exact task2 geometric augmentation`.

### Task 3: Representación común CNN mínima

**Files:**
- Create: `src/fido/models/task2_common.py`
- Create: `src/fido/losses/task2_contrastive.py`
- Create: `src/fido/train/train_task2_common.py`
- Create: `src/fido/tests/test_task2_correspondence_loss.py`
- Modify: `analysis/measure_task2_correspondence.py`
- Create: `experiments/82-t2-common-cnn/`

**Interfaces:**
- Produces: `FundusDenseEncoder`, `OctDenseEncoder` y descriptores L2-normalizados.
- Produces: `sample_positive_pairs(fundus_desc, oct_desc, gt_matrix, valid_mask, k=64)`.
- Produces: `dense_infonce(positive, negatives, temperature) -> Tensor`.

- [ ] **Step 1: Tests de warp GT y correspondencia sintética**

```python
def test_positive_sampler_recovers_known_translation():
    pairs = sample_positive_pairs(fundus_desc, oct_desc, translated_gt(32, 48), valid, k=16)
    assert torch.allclose(pairs.fundus_xy - pairs.oct_xy, torch.tensor([32., 48.]), atol=1.)

def test_infonce_prefers_true_positive():
    assert dense_infonce(pos_same, neg_orthogonal, 0.1) < dense_infonce(pos_wrong, neg_orthogonal, 0.1)
```

- [ ] **Step 2: Ejecutar tests y confirmar fallo**

- [ ] **Step 3: Implementar encoders separados y loss densa**

Dimensión inicial 128, K=64, negativos intraimagen y exclusión de radio 12 px. Sin cabeza de pose, DINO, hard negatives cross-frame ni proyección nueva.

- [ ] **Step 4: Entrenar gate corto fold 0**

Run: `PYTHONPATH=src python -m fido.train.train_task2_common --root /workspace/data/Task2 --enface-cache /root/data_cache/Task2_enface --encoder cnn --fold 0 --seed 0 --epochs 10 --k 64 --out checkpoints/task2_common_cnn`

- [ ] **Step 5: Evaluar representación**

GO si M1 top-1% `>=50%`, M2 con pose parcial GT `>=50%` a 10 px, mediana `<37.12 px` y OCT-shuffle reduce recall al menos 30 puntos. Si falla, no construir solver.

- [ ] **Step 6: Commit**

Commit sugerido: `feat: learn dense task2 cross-modal descriptors`.

### Task 4: Proyección axial fija multicanal

**Files:**
- Create: `src/fido/data/task2_projection.py`
- Modify: `src/fido/data/common.py`
- Modify: `src/fido/data/task2.py`
- Modify: `infra/precompute_task2_enface.py`
- Create: `src/fido/tests/test_task2_projection.py`
- Create: `experiments/83-t2-fixed-projections/`

**Interfaces:**
- Produces: `fixed_oct_projections(volume) -> Tensor[C,S,W]` con mean, MIP, min, std, gradiente axial y slabs.
- Cache versionado con nombre de esquema y dtype explícito.

- [ ] **Step 1: Tests de shape, determinismo y ejes**

```python
def test_projection_preserves_lateral_axes():
    out = fixed_oct_projections(torch.zeros(128, 512, 512))
    assert out.shape[-2:] == (128, 512)
```

- [ ] **Step 2: Ejecutar test y confirmar fallo**

- [ ] **Step 3: Implementar proyecciones y cache versionado**

- [ ] **Step 4: Repetir únicamente Task 3 con nuevo input**

- [ ] **Step 5: Gate**

GO si M1 sube 10 puntos o M2 mediana baja 20%, con overhead `<=0.15 s/caso`; si falla volver a mean.

- [ ] **Step 6: Commit**

Commit sugerido: `feat: add multichannel oct projections`.

### Task 5: Hard negatives del mismo escenario

**Files:**
- Create: `src/fido/data/task2_sampler.py`
- Modify: `src/fido/losses/task2_contrastive.py`
- Modify: `src/fido/train/train_task2_common.py`
- Modify: `src/fido/tests/test_task2_correspondence_loss.py`
- Create: `experiments/84-t2-hard-negatives/`

**Interfaces:**
- Produces: batches pareados con negativos de otros frames del mismo escenario.
- Excluye falsos negativos a `<24 px` de la correspondencia GT.

- [ ] **Step 1: Testear que sampler no cruza escenario y respeta exclusión**

- [ ] **Step 2: Implementar sampler sin cambiar encoder/proyección**

- [ ] **Step 3: Entrenar fold 0**

- [ ] **Step 4: Gate**

GO si margen positivo-hard-negative mejora 20%, M1/M2 suben 5 puntos y paired vs OCT-shuffle difiere 30 puntos.

- [ ] **Step 5: Commit**

Commit sugerido: `feat: mine same-scenario task2 negatives`.

### Task 6: DINOv2 dual con LP-FT y LLRD

**Files:**
- Modify: `src/fido/models/task2_dinov2.py`
- Create: `src/fido/train/task2_dinov2_optim.py`
- Modify: `src/fido/train/train_task2_common.py`
- Create: `src/fido/tests/test_task2_dinov2_dual.py`
- Create: `experiments/85-t2-dino-lpft/`

**Interfaces:**
- Produces: dos instancias DINOv2 independientes, adaptadores por modalidad y mismo descriptor/loss que CNN.
- Produces: param groups LLRD por rama.

- [ ] **Step 1: Tests de no-sharing y LR**

```python
def test_dual_dino_does_not_share_weights():
    model = DualDinoCommonEncoder(pretrained=False)
    assert model.fundus.blocks[0].attn.qkv.weight.data_ptr() != model.oct.blocks[0].attn.qkv.weight.data_ptr()
```

- [ ] **Step 2: Ejecutar tests y confirmar fallo**

- [ ] **Step 3: Implementar LP de tres épocas**

Backbones congelados; solo adaptadores/proyección común entrenables.

- [ ] **Step 4: Implementar unfreeze y LLRD**

Descongelar últimos cuatro bloques como screening, heads `1e-4`, último bloque `1e-5`, decay `0.75`, warmup 5%, cosine. Full unfreeze posterior es otro rung solo si partial FT pasa.

- [ ] **Step 5: Entrenar y perfilar**

Run: `PYTHONPATH=src python -m fido.train.train_task2_common --root /workspace/data/Task2 --encoder dinov2 --stage lpft --fold 0 --seed 0 --lp-epochs 3 --ft-epochs 10 --head-lr 1e-4 --last-block-lr 1e-5 --layer-decay 0.75 --out checkpoints/task2_common_dino`

- [ ] **Step 6: Gate**

GO si supera CNN al menos 5 puntos M2 o reduce mediana 15%, paired/OCT-shuffle sigue separándose y ambos encoders cuestan `<3 s/caso`.

- [ ] **Step 7: Commit**

Commit sugerido: `feat: fine-tune dual dinov2 task2 encoders`.

### Task 7: Búsqueda angular con reflexión fija

**Files:**
- Create: `src/fido/matching/__init__.py`
- Create: `src/fido/matching/task2_pose_search.py`
- Create: `src/fido/tests/test_task2_pose_search.py`
- Create: `analysis/evaluate_task2_pose_components.py`
- Create: `experiments/86-t2-angle-search/`

**Interfaces:**
- Produces: `search_rotation(template, search, num_bins=64, reflect=True) -> PoseCandidates`.

- [ ] **Step 1: Testear ángulos sintéticos, wraparound y reflexión**

```python
def test_angle_search_recovers_wrapped_rotation():
    candidates = search_rotation(template, rotate_reflect(template, 179.0), 64, True)
    assert circular_error(candidates.best.theta_deg, 179.0) < 3.0
```

- [ ] **Step 2: Ejecutar test y confirmar fallo**

- [ ] **Step 3: Implementar 64 bins batched y refine +/-1 bin**

Para este rung posición y escala son GT. No añadir búsqueda multiescala.

- [ ] **Step 4: Gate**

GO si mediana angular `<3 deg`, p90 `<6 deg` y solver `<1 s/caso`.

- [ ] **Step 5: Commit**

Commit sugerido: `feat: add reflected task2 rotation search`.

### Task 8: Búsqueda multiescala y traslación

**Files:**
- Modify: `src/fido/matching/task2_pose_search.py`
- Modify: `src/fido/tests/test_task2_pose_search.py`
- Modify: `analysis/evaluate_task2_pose_components.py`
- Create: `experiments/87-t2-pose-search/`

**Interfaces:**
- Produces: `search_similarity(template, search, scale_bins, angle_bins, top_k) -> PoseCandidates`.
- Scale bins se derivan solo de train y cubren una extrapolación física pre-registrada.

- [ ] **Step 1: Tests sintéticos de escala, traslación y top-K**

- [ ] **Step 2: Implementar búsqueda coarse-to-fine en chunks**

Codificar cada modalidad una vez; conservar top-3 escalas y refinar log-scale continuo.

- [ ] **Step 3: Evaluar holdout sintético de escala**

- [ ] **Step 4: Gate**

GO si escala mediana `<=3%`, p90 `<=6%`, centro `<=10 px` en `>=50%`, AUC fold `>0.10` y encoder+solver `<=4 s/caso`.

- [ ] **Step 5: Commit**

Commit sugerido: `feat: add multiscale task2 pose search`.

### Task 9: Top-K y Umeyama reflejado

**Files:**
- Modify: `src/fido/geometry.py`
- Create: `src/fido/matching/task2_topk.py`
- Create: `src/fido/tests/test_task2_umeyama.py`
- Create: `src/fido/tests/test_task2_topk.py`
- Create: `experiments/88-t2-topk-umeyama/`

**Interfaces:**
- Produces: `fit_reflected_similarity_umeyama(src_xy, dst_xy, weights) -> Tensor[3,3]`.
- Produces: NMS top-8 y mutual descriptor checks.

- [ ] **Step 1: Tests exactos con ruido cero, outliers y pesos**

```python
def test_reflected_umeyama_recovers_exact_matrix():
    estimated = fit_reflected_similarity_umeyama(src, project(m, src), torch.ones(len(src)))
    assert torch.allclose(estimated, m, atol=1e-4)
    assert torch.det(estimated[:2, :2]) < 0
```

- [ ] **Step 2: Ejecutar tests y confirmar fallo**

- [ ] **Step 3: Implementar solver 4-DOF; prohibido DLT 8-DOF**

- [ ] **Step 4: Añadir top-K, mutual check y reponderación robusta**

- [ ] **Step 5: Gate**

GO si mediana o p90 baja 25%, AUC sube `>=0.05` y no aumentan errores `>100 px`.

- [ ] **Step 6: Commit**

Commit sugerido: `feat: fit robust reflected task2 similarity`.

### Task 10: Refinador local condicionado

**Prerequisite:** Task 8/9 produce AUC `>0.10` y mediana `<25 px`.

**Files:**
- Create: `src/fido/models/task2_refiner.py`
- Create: `src/fido/train/train_task2_refiner.py`
- Create: `src/fido/tests/test_task2_refiner.py`
- Create: `experiments/89-t2-local-refiner/`

**Interfaces:**
- Produces: updater compartido para `dtx,dty,dtheta,dlog_scale` durante 4–6 iteraciones.

- [ ] **Step 1: Testear identidad, convergencia sintética y límites**

- [ ] **Step 2: Construir dataset de residuos out-of-fold reales**

- [ ] **Step 3: Entrenar refinador sin tocar globalizador**

- [ ] **Step 4: Gate**

GO si mediana baja 25%, p90 no empeora más de 5%, AUC sube `>=0.05` y overhead `<0.5 s/caso`; si no, omitir del artefacto.

- [ ] **Step 5: Commit**

Commit sugerido: `feat: add conditional task2 local refinement`.

### Task 11: Confirmación OOF y selección final

**Files:**
- Create: `analysis/evaluate_task2_pipeline.py`
- Create: `src/fido/tests/test_task2_pipeline.py`
- Create: `experiments/90-t2-oof-selection/`

**Interfaces:**
- Produces: AUC pooled/per-scenario, CDF@0..10, errores de componentes, p95 y OCT-shuffle.

- [ ] **Step 1: Confirmar fold 0 winner en folds 1 y 2**

- [ ] **Step 2: Si pasa, ejecutar cinco folds/OOF**

- [ ] **Step 3: Gate final de arquitectura**

Mediana OOF `>=0.20`, al menos 3/5 folds `>0.10`, OCT-shuffle colapsa y mejora no depende de un solo escenario.

- [ ] **Step 4: Entrenar receta congelada sobre diez escenarios**

Epochs = mediana de mejores epochs CV; no mirar Mock.

### Task 12: Empaquetado, fallback y perfil end-to-end

**Files:**
- Create: `analysis/profile_task2_inference.py`
- Create: `submissions/r04-task2-common/inference.py`
- Create: `submissions/r04-task2-common/README.md`
- Create: `src/fido/tests/test_task2_submission_contract.py`

**Interfaces:**
- Produces: `model_1.pth`, zip plano y matriz fallback determinista.

- [ ] **Step 1: Testear firma, `None`, shape, dtype y fila homogénea**

```python
def test_submission_returns_valid_matrix_without_oct():
    m = inference(1, None, np.zeros((1024, 1024, 3), np.uint8), model)
    assert m.shape == (3, 3) and m.dtype == np.float64
    assert np.allclose(m[2], [0, 0, 1]) and np.isfinite(m).all()
```

- [ ] **Step 2: Implementar inferencia autocontenida**

No declarar `torch` ni `numpy` en requirements.

- [ ] **Step 3: Ejecutar suite y perfil de 100 casos**

```bash
PYTHONPATH=src python -m pytest -q src/fido/tests
PYTHONPATH=src python analysis/profile_task2_inference.py --cases 100 --include-io --container-mode
```

Expected: cero excepciones/timeouts, ningún caso `>20 s`, total `<=540 s`.

- [ ] **Step 4: Congelar hashes y ejecutar una vez el Mock agregado**

Run: `python eval/run_local.py --submission submissions/r04-task2-common --task registration --save`

- [ ] **Step 5: Inspeccionar zip plano y registrar submission**

Commit sugerido fuera de master: `feat: package cross-modal task2 submission`.

## Calendario y asignación de GPU

- 18–21 agosto: Tasks 1–3; diagnósticos y representación mínima. L4 para gates cortos.
- 21–25 agosto: Tasks 4–6; proyección, hard negatives y DINO LP-FT solo si el gate común funciona.
- 25–29 agosto: Tasks 7–9; solver y top-K.
- 29–30 agosto: Task 10 solo si cumple prerequisite.
- 30 agosto: congelar arquitectura.
- 31 agosto–3 septiembre: folds, entrenamiento final, empaquetado, perfil y contingencia.
- RTX 5090 se reserva para entrenamientos aprobados; Task 2 mantiene prioridad constitucional. Estimación condicionada por gates: 55–110 GPU-h.
- Stop: si la representación no supera Task 3/4, no construir solver grande; entregar el mejor control reproducible y preservar el tiro final.
