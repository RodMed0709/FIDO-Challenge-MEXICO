# T2-80 — diagnóstico de componentes y representación

## Hipótesis

El error del checkpoint `task2_fixed/model_1.pth` está dominado por posición y
escala, mientras que sus features densos no forman todavía un espacio común
fundus–OCT. Sustituir cada componente por GT y mezclar el OCT entre casos del
mismo fold permiten separar ambas afirmaciones sin cambiar el modelo.

## Corrida congelada

- Dataset: Task 2 de entrenamiento únicamente; nunca Mock Test por caso.
- Split: `GroupKFold(n_splits=5)`, fold 0, seed 0.
- Checkpoint: `/workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth`.
- Provenance adversarial: este checkpoint usa `SCALE_REF=160`, constante
  obtenida antes de imponer el split por escenarios. Por tanto T2-80 lo trata
  como baseline **transductivo/no limpio**, útil para diagnóstico pero no como
  evidencia OOF autoritativa. T2-81 deberá derivar todo rango/referencia solo de
  `train_idx` y demostrar que cambios extremos en validación no lo alteran.
- Representación: K=64 puntos GT por caso; exactamente los mismos índices de
  validación y semilla para paired y OCT-shuffle.
- Métrica oficial: error medio de cuatro esquinas y AUC en umbrales enteros
  0..10, usando `fido.geometry.corner_error/corner_auc`.

## Sustituciones independientes pre-registradas

1. Posición perfecta: sustituir `tx,ty`, conservar rotación y escala predichas.
2. Rotación perfecta: sustituir `cos,sin`, conservar posición y escala predichas.
3. Escala perfecta: sustituir `scale`, conservar posición y rotación predichas.

Se reportarán baseline, las tres sustituciones, combinaciones como controles,
error/AUC pooled y por escenario, y un JSONL con una fila por caso y variante.
La prioridad posterior será el componente cuya sustitución produzca la mayor
subida de AUC; el diagnóstico no autoriza una submission.

## Expectativas y gates

- Control todo-GT: error numérico cercano a cero y AUC 1.0 para matrices de
  similitud puras; con GT real recompuesto se acepta el residuo 4-DOF ya medido.
- M1 paired: baseline conocido top-1% 0.34% (vecino) / 0.39% (bilineal).
- M2 paired: recall a 10 px 6.9%, mediana 37.12 px.
- M3 paired: masa mediana 0.090954, correlación -0.017.
- OCT-shuffle debe degradar M1 y M2 respecto a paired. Si no lo hace, el modelo
  no demuestra usar contenido OCT específico y no se autoriza construir solver.
- Gate para el siguiente modelo común: M1 top-1% >=50%, M2 <=10 px >=50%,
  mediana <37.12 px, y caída por shuffle de al menos 30 puntos porcentuales.

## Comandos reproducibles (pod, no ejecutar en este rung local)

```bash
PYTHONPATH=src python analysis/diagnose_task2_error.py \
  --root /workspace/data/Task2 --enface-cache /root/data_cache/Task2_enface \
  --checkpoint /workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth \
  --fold 0 --device cuda \
  --output experiments/80-t2-diagnostics/component_metrics.json \
  --jsonl experiments/80-t2-diagnostics/component_cases.jsonl

PYTHONPATH=src python analysis/measure_task2_correspondence.py \
  --root /workspace/data/Task2 \
  --checkpoint /workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth \
  --output experiments/80-t2-diagnostics/correspondence.md \
  --jsonl experiments/80-t2-diagnostics/correspondence_cases.jsonl \
  --n-folds 5 --fold 0 --k 64 --seed 0 --device cuda --oct-shuffle
```
