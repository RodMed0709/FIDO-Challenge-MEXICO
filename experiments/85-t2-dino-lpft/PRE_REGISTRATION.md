# T2-85 — dual DINOv2 LP→partial-FT

## Condición de apertura

No ejecutar hasta que T2-83 tenga `gate_pass=true` y exista un `results.json`
T2-82 repetido bajo la convención canónica. Ese artefacto se pasa mediante
`--control-results`; Mock Test permanece prohibido.

## Hipótesis única

Sustituir únicamente los encoders CNN por dos DINOv2 ViT-S/14 independientes,
primero como linear probe y después adaptando sus cuatro últimos bloques,
mejora la correspondencia densa. Proyección, convención, GroupKFold,
augmentación T2-81, K=64, negativos, InfoNCE y evaluación permanecen idénticos.

## Arquitectura y fases congeladas

- Ramas fundus/OCT independientes; adaptador OCT 1→3.
- Features de bloques 2/5/8/11, FPN ligero, descriptor 128 L2.
- Upsampling 2x: stride efectivo 7.
- LP 3 épocas con backbones congelados.
- Partial FT 10 épocas, solo últimos cuatro bloques; heads `1e-4`, último
  bloque `1e-5`, LLRD `0.75`, warmup 5%, cosine.
- BF16, acumulación 4, grad clip 1.0. Full FT explícitamente fuera de alcance.
- Fundus completo se remuestrea 1024→1022: `73×73=5329` tokens antes del FPN;
  OCT canónico completo se remuestrea a 224²: `16×16=256` tokens. No crops.
- Fallback `oct_volume=None` permanece una matriz nativa determinista y la
  exportación usa `task2_canonical_v1`; ninguna de ambas se selecciona con Mock.

```bash
PYTHONPATH=src python -m fido.train.train_task2_common \
  --root /workspace/data/Task2 --enface-cache /root/data_cache/Task2_enface \
  --encoder dinov2 --stage lpft --fold 0 --seed 0 \
  --batch-size 1 \
  --lp-epochs 3 --ft-epochs 10 --head-lr 1e-4 \
  --last-block-lr 1e-5 --layer-decay 0.75 --grad-accum 4 \
  --amp-dtype bfloat16 --k 64 \
  --control-results checkpoints/task2_common_cnn/results.json \
  --control-checkpoint checkpoints/task2_common_cnn/model_1.pth \
  --t2-83-gate experiments/83-t2-enface-convention/oracle_train.json \
  --out checkpoints/task2_common_dino
```

Antes de la corrida completa se ejecuta el mismo comando con
`--smoke-valid-steps 2`; debe completar dos optimizer steps válidos, escribir
peak CUDA memory y no producir OOM. Ese smoke no es resultado científico.

`--resume` solo acepta checkpoints de fin de época con receta
`task2_common_recipe_v1` exactamente idéntica: LRs, LLRD, acumulación, AMP,
batch/input sizes, épocas, K, temperatura, negativos, augmentación y número de
steps del scheduler. Cualquier diferencia aborta antes de cargar pesos.

Los prerequisitos son fail-closed: se recalculan hashes de IDs train/val desde
el dataset actual, se exige `task2_enface_convention_gate_v1`, C exacta, y se
abre el checkpoint control cuyo SHA figura en `results.json` para validar su
frame, schema, fold, seed, val IDs, hashes y receta interna.

## Gate

Debe conservar el gate absoluto T2-82, mejorar M2 >=5 puntos o reducir su
mediana >=15%, mantener separación OCT-shuffle, reportar pooled/per-scenario y
medir encoders `<3 s/caso`. Cualquier fallo es NO-GO. Full FT requiere otro
pre-registro.
