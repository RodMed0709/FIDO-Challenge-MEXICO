# 40 — Task 2: quitar el techo estructural del decodificador

Peldaño de la escalera: **T2-R2, versión corregida** (`ATTACK_LADDER.md`).

## Escalera

| run | qué es | métrica |
|---|---|---|
| `../30-t2-baseline-roto/` 30a | control: mismo modelo, mismo split, mismo fold | `corner_auc` **0.0000**, error 98.74 px |
| **40a** | 3 bugs estructurales corregidos + pérdidas rebalanceadas, 60 épocas | **`corner_auc` 0.0048**, error 66.25 px |

## Expectativa pre-registrada

Con el techo estructural eliminado, el AUC debería dejar de ser exactamente cero.
No se pre-registró una cifra concreta porque la magnitud dependía de cuánto del
error observado era el bug y cuánto era aprendizaje real — precisamente lo que
este peldaño mide.

## 🟢 RESULTADO (2026-08-17) — primer AUC no-cero del proyecto en Task 2

| época | `val_mean_error` | `val_corner_auc` |
|---|---|---|
| 5 | 110.97 px | 0.0000 |
| 15 | 89.52 px | 0.0000 |
| 25 | 74.48 px | **0.0018** ← primer no-cero |
| 45 | 57.81 px | 0.0033 |
| 60 | 66.25 px | **0.0048** (mejor) |

**Costo**: ~$1.5 de pod (60 épocas en ~40 min, contra 1h14m que costaban 25
épocas antes de cachear el en-face).

## La prueba que justifica el peldaño

Alimentando el decodificador con localización y parámetros **perfectos** (sacados
del GT, sin ninguna predicción) — es decir, midiendo el techo estructural:

| decodificador | error de esquina | AUC |
|---|---|---|
| viejo (30a) | 112.88 px | **0.0000** |
| nuevo (40a) | 1.81 px | **0.7848** |

El código anterior era incapaz de puntuar aunque entrenara perfecto. Los 1.81 px
residuales son redondeo del round-trip `decompose→compose`, no error real.

## Los tres bugs estructurales

1. **Centro vs esquina.** El pico de la correlación cruzada marca dónde se alinea
   el *centro* de la plantilla; el código lo asignaba a `(tx,ty)`, que en el GT es
   la esquina `uv=(0,0)`. Separación medida sobre 1214 casos: **113.1 ± 9.9 px**
   (teórica `s·√2/2` con s=159.6: 112.9 px).
   *Ironía*: el propio comentario del modelo decía "esto YA ES el centro" y dos
   líneas después lo trataba como esquina.
   Fix: `tx = cx - s·(cos+sin)/2`, `ty = cy - s·(sin-cos)/2`, y el heatmap GT
   generado en el centro, no en la esquina.

2. **Sesgo de media celda.** Kernel de correlación con lados PARES y
   `padding=(Ht//2, Wt//2)`: la celda de salida `o` corresponde al centro `o-0.5`.
   Son 0.5 celdas × 16 px = 8 px por eje → **11.3 px de error de esquina, por
   encima del umbral de 10 px del AUC por sí solo**.
   Verificado incrustando una plantilla conocida: sin corrección el pico sale a
   (+0.50, +0.50) celdas; con `-0.5`, exactamente 0.00.

3. **Plantilla 4:1 contra huella cuadrada.** El volumen OCT cubre una región
   cuadrada de retina (el GT es una similitud, ortogonalidad 100%) pero se
   muestrea anisótropo: 128 slices × 512 A-scans → plantilla de 8×32 celdas,
   cuando la huella real es ~160×160 px = 10×10 celdas. La correlación comparaba
   geometrías incompatibles.
   Fix: remuestrear el en-face a `SCALE_REF²` antes del encoder.

## Defectos de optimización corregidos en el mismo pase

- `scale = softplus(raw)+1` arrancaba en **1.69** contra un objetivo de ~160 → el
  término de escala valía ~45, el **97% de la pérdida total**, y monopolizaba el
  presupuesto de `clip_grad_norm_(1.0)`. Ahora `scale = 160·exp(raw)` y el MSE en
  log.
- `coord_l1` en píxeles crudos daba normas de gradiente de miles contra un clip de
  1.0 — el clip actuaba como learning rate fijo minúsculo. Normalizado por el
  tamaño de imagen.
- El BCE del heatmap con reducción media dejaba la señal de localización en el
  1.6% del mapa (69 de 4225 celdas), o sea en el ruido. Ahora con `pos_weight=50`
  y peso 1.0 en vez de 0.1.
- Selección de checkpoint con desempate por error medio (ver peldaño 30).

## Veredicto honesto

**0.0048 no es competitivo.** El líder del leaderboard va en 0.475. Lo que este
peldaño consiguió es quitar el suelo de cristal: el modelo ahora sí responde al
entrenamiento. El error se aplana ruidoso entre 57 y 72 px.

## Siguiente límite, ya identificado y medido

La rotación se predice desde un vector de features **promediado globalmente**
(`mean(dim=(2,3))`), y el promedio destruye justo la información de orientación.
El error de esquina por rotación es `≈ s·Δθ`, así que bajar de 10 px exige
`Δθ < 3.6°`. Con el error actual de 66 px, si todo fuera rotación serían ~24° de
error angular.

**Propuesta para el peldaño 60**: en vez de regresar el ángulo, **buscarlo**:
rotar el mapa de features del en-face en N ángulos discretos, correlacionar cada
uno, y tomar el argmax sobre (ángulo, posición). Convierte la estimación de
rotación en una búsqueda, que es lo que la literatura de DPCN++/log-polar hace.
Los encoders corren una sola vez; solo se repite la correlación, que es barata.

## Cómo reproducir

```bash
python -m fido.train.train_task2_baseline \
  --root /workspace/data/Task2 \
  --enface-cache /root/data_cache/Task2_enface \
  --epochs 60 --batch-size 8 --lr 3e-4 --n-folds 5 --fold 0 --val-every 5 \
  --out checkpoints/task2_fixed
```

**`--enface-cache` no es opcional en la práctica**: sin él cada caso abre los 128
PNG del volumen (812.7 ms/caso contra 20.7 ms). Ver `HANDOFF.md`.
