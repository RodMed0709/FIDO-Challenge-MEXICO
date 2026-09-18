# 00 — smoke del contrato: ¿podemos entregar algo que produzca un número?

## Escalera

| run | qué es | Task 1 | Task 2 |
|---|---|---|---|
| `r00a_smoke` | `inference.py` trivial, pesos placeholder | **0.000000** | **0.000000** |

## Expectativa pre-registrada

**Score esperado: 0.** Este peldaño no prueba ningún modelo. Prueba que el
**contrato de entrega** funcione de punta a punta, que es la falla que sospecho
detrás de los cuatro `0.000` del leaderboard de Task 2.

Lo que se verifica:
- firma `inference(task_id, oct_volume, opmi_image, model)` de **4** argumentos
  (la documentación de Codabench muestra 3, y está mal)
- nombres `model_0.pth` / `model_1.pth`, no `model.pth`
- manejo de `oct_volume=None` sin reventar la corrida completa
- formatos de retorno: dict para Task 1, array `(3,3)` float64 para Task 2

**Criterio de éxito: que salga un NÚMERO en vez de un error de ingestión.**

## 🟢 RESULTADO (2026-08-14) — el contrato funciona

```
  keypoints    :  score = 0.000000   casos = 5   timeouts = 0
                  keypoint_auc = 0.0   distance_auc = 0.0
  registration :  score = 0.000000   casos = 5   timeouts = 0
```

Los 4 puntos del contrato quedaron validados en la práctica, no en teoría.

**Hallazgo secundario, y vale**: el run de Task 2 usó una similitud con escala
fija y rotación cero — o sea, un predictor esencialmente constante. Dio
**exactamente 0.0**. Eso confirma por medición que en Task 2 **no existe atajo
trivial**: la matriz varía demasiado entre casos como para que una constante
puntúe algo.

**Costo: $0** (corrido en local contra el Mock Test, sin GPU).

## Por qué existe este peldaño

Hay 20 submissions en toda la fase Competition. Quemar una descubriendo que el
peso se llamaba `model_0.pth` y no `model.pth` sería inaceptable. Este peldaño
compra esa certeza a costo cero.

Además establece la **regla de oro** del proyecto: nada sube a Codabench sin
pasar antes por `eval/run_local.py`, que importa la ingestión y el scoring
vendorizados **sin modificarlos**.

## Cómo reproducir

```bash
python eval/run_local.py --submission experiments/00-smoke-contract/runs/r00-smoke --task both
```
