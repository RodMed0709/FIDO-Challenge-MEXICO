# T2-83 — convención en-face canónica

## Hipótesis única

La proyección nativa `(slice,lateral)` está orientada respecto al GT mediante
`transpose__flip_u__flip_v`. Materializar esa única corrección antes del
encoder elimina supervisión espacial contradictoria y debe recuperar el
enriquecimiento de instrumento observado por el oráculo.

No se cambian proyección de intensidad, arquitectura, loss, negativos ni
decoder. La implementación única es `canonicalize_task2_enface`; tanto cache
como lectura cruda llegan a ella en `Task2Dataset`.

La matriz JSON está en el frame nativo. Con
`C=[[0,-1,1],[-1,0,1],[0,0,1]]` (canonical→native), el pipeline usa
`M_canonical=M_native@C`. Como `det(C)=-1`, la similitud interna es propia
(`reflect=False`). Toda augmentación fundus posterior conserva
`M_aug=H@M_canonical`; una predicción final vuelve al contrato oficial mediante
`M_native=M_canonical@C` porque `C=C^-1`.

## Protocolo y gate

- Task 2 training solamente; GroupKFold 5, fold 0, seed 0.
- Mock Test está prohibido.
- El cache conserva orientación nativa y declara
  `native_slice_lateral_v1`; el dataset aplica la corrección exactamente una vez.
- GO si hay >=100 puntos de instrumento y, en radios 0.01/0.02, la convención
  esperada logra ratio observado/null >=2.0 y >=1.5x el control identity.

```bash
PYTHONPATH=src:. python analysis/validate_task2_enface_convention_train.py \
  --root /workspace/data/Task2 --n-folds 5 --fold 0 --seed 0 --workers 32 \
  --output experiments/83-t2-enface-convention/oracle_train.md \
  --json experiments/83-t2-enface-convention/oracle_train.json
```

Solo después de `gate_pass=true` se permite repetir el entrenamiento T2-82 con
la convención corregida, manteniendo su comando e hiperparámetros congelados.
