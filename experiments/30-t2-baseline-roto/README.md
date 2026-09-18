# 30 — Task 2 baseline: la corrida que no podía puntuar

Peldaño de la escalera: **T2-R2, primera versión** (`ATTACK_LADDER.md`).

> **Este peldaño existe como cicatriz, no como resultado.** Se conserva porque
> explica por qué el peldaño 40 fue necesario, y porque la conclusión que se
> escribió al cerrarlo era **incorrecta** — un error de razonamiento que vale
> más documentar que esconder.

## Escalera

| run | qué es | métrica |
|---|---|---|
| `../00-smoke-contract/` | control: predictor constante | `corner_auc` 0.0000 |
| **30a** | heatmap por correlación cruzada + regresión de (θ,s), 25 épocas | **`val_corner_auc` 0.0000**, `val_mean_error` 98.74 px |

## Expectativa pre-registrada

`corner_auc` 0.20-0.35. Sirve de piso real (no trivial) contra el que medir todo
lo demás.

## 🔴 RESULTADO (2026-08-17) — AUC 0.0000

Sobre el dataset completo, fold 0 de 5 (966 train / 248 val):

| época | `val_mean_error` | `val_corner_auc` |
|---|---|---|
| 5 | 203.29 px | 0.0000 |
| 10 | 220.64 px | 0.0000 |
| 15 | 107.85 px | 0.0000 |
| 20 | 111.65 px | 0.0000 |
| 25 | **98.74 px** | 0.0000 |

**Costo**: ~$2 de pod + 3 relanzamientos por problemas de I/O.

## ⚠️ La conclusión original era falsa

Al cerrar esta corrida se escribió: *"no es un bug, es techo de capacidad/tiempo
de entrenamiento, no de diseño roto"*, razonando que el smoke test de overfit ya
se estancaba en 165-200 px con solo 5 casos.

**Era incorrecto.** El peldaño 40 demostró que el modelo tenía tres defectos
estructurales, y que el principal (confundir el centro de la plantilla con la
esquina del cuadrado unitario) ponía un **techo duro de AUC en 0.031** — o sea,
esta corrida no podía puntuar aunque hubiera entrenado perfectamente. El
estancamiento del smoke test no era falta de capacidad: era la firma del bug.

Los 98.74 px de error coinciden, dentro del ruido, con los **113.1 px** de offset
geométrico centro↔esquina medidos sobre los 1214 casos reales.

## Lección

Un estancamiento no es evidencia de "techo de capacidad". Antes de aceptar esa
explicación hay que medir el **techo estructural**: alimentar el decodificador
con la respuesta perfecta (tomada del GT) y ver qué score sale. Si con entrada
perfecta el score ya es malo, el problema no está en el aprendizaje.

Esa prueba cuesta minutos y aquí habría ahorrado ~$2 de GPU y una conclusión
equivocada escrita en tres archivos.

## Checkpoint

`model_1.pth` guardado corresponde a la **época 5** (el peor de los cinco
evaluados), no a la 25. Causa: `if auc > best_auc` con `best_auc = -1.0` y el AUC
saturado en 0.0000 — la primera validación entra y ninguna posterior puede
superarla. Corregido en el peldaño 40 con desempate por error medio.

Respaldado en el pod: `/workspace/checkpoints_backup/model_1_ep5_err203px.pth`
