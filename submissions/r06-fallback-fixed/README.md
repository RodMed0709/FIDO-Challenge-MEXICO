# r06 — fallback de distancia reescalado

> **Nota de corrección (2026-09-18).** Este README se dejó como copia literal del
> de `r05` y por eso el texto de abajo se titula y se describe como r05. El cambio
> real de `r06` **no** está descrito ahí: es una sola constante en `inference.py`,
> `DISTANCE_FALLBACK_PX 159.2 -> 203.8` (el mismo factor 1.28 que ya se había
> aplicado a `A` y `B` en r05). Los pesos son bit-idénticos a r04 y r05.
>
> Motivo: 159.2 era la constante óptima medida contra el target viejo (`GT/10`).
> Con el target nuevo (`GT/7.8125`) queda 45.6 px por debajo del centro de la
> distribución, muy por encima del umbral de 20 px — o sea, **cada caso que cae en
> el fallback puntuaba cero garantizado**. Afecta al ~10% de casos sin OCT más
> aquellos donde la segmentación no logra medir el gap.
>
> Score real en Codabench: **0.5691**, con `keypoint_auc = 0.7573`.
>
> El texto original de r05 se conserva íntegro debajo, sin editar.

---

# r05 — distancia reescalada

Parte de `r04-interim-joint` **sin tocar los pesos** (`model_0.pth` y
`model_1.pth` son bit-idénticos a r04, verificado con `diff`). El único
cambio es en `inference.py`: las dos constantes del ajuste lineal
`pixel_gap -> distancia` se multiplican por `1.28` para seguir la
resolución de profundidad corregida por los organizadores el 2026-08-19
(`GROUND_TRUTH_DISTANCE_SCALE`: `10 -> 7.8125`; `MAX_THRESHOLD_DIST`:
`10 -> 20`, ambas ya en `vendor/fido/Codabench Bundle/scoring_program/
scoring_keypoints.py`).

| Constante | r04 (heredado de r01/r03) | r05 |
|---|---:|---:|
| `DISTANCE_SCALE_A` | 0.7320 | **0.9370** (`= 0.7320 * 1.28`) |
| `DISTANCE_SCALE_B` | 2.6779 | **3.4277** (`= 2.6779 * 1.28`) |
| `DISTANCE_FALLBACK_PX` | 159.2 | 159.2 (**sin cambiar**, ver abajo) |

**Por qué la multiplicación por 1.28 es exacta, no una aproximación**: el
ajuste `A, B` original es una regresión OLS de `pixel_gap -> GT/10`
(`analysis/verify_task1_distance_geometry.py`, T1-R2, R²=0.9916 sobre
92,012 mediciones). El nuevo target es `GT/7.8125 = (GT/10)·1.28`, es decir
el mismo target escalado por una constante `k=1.28`. Para OLS, si
`Y_new = k·Y_old`, la solución óptima es exactamente `a_new = k·a_old`,
`b_new = k·b_old` — no es un nuevo ajuste aproximado, es álgebra de mínimos
cuadrados bajo reescalado lineal del target. `R²` es invariante a esa
transformación, así que sigue siendo 0.9916. Detalle completo en
`experiments/86-t1-margen/INFORME.md`, sección 1.

**`DISTANCE_FALLBACK_PX` deliberadamente sin tocar**: es la mejor constante
posible cuando no hay medición geométrica (10% de casos sin OCT, o
segmentación sin punta/ILM visibles), optimizada para AUC bajo el umbral y
target VIEJOS (`0.0445` sobre 61,691 casos a `MAX_THRESHOLD_DIST=10`,
target `GT/10`). No es una regresión OLS, así que la propiedad de
reescalado exacto no aplica: el óptimo bajo el umbral nuevo (0..20) y
target nuevo (`GT/7.8125`) no tiene por qué ser `1.28×` este valor (el
umbral escaló `2×`, no `1.28×`), y recalcularlo correctamente exige el
dataset completo de train (solo disponible en el pod). Tocarlo sin esa
medición sería la clase de "mejora extra sin poder atribuir causa" que este
peldaño evita a propósito. Afecta solo al subconjunto sin medición
geométrica; ver `experiments/86-t1-margen/INFORME.md` sección 1 y 4.

## Contrato de submission verificado

- Firma `inference(task_id, oct_volume, opmi_image, model)` — sin cambios,
  heredada de r04 (`inference.py:462`).
- Pesos `model_0.pth` / `model_1.pth` en la raíz del paquete — presentes.
- `requirements.txt` — no incluido (no se declaran `numpy`/`torch`, ya
  instalados en el evaluador, regla 4 de `RULES_OF_ENGAGEMENT.md`).
- `oct_volume is None` manejado (`_infer_task1`/`_infer_task2` comprueban
  `oct_volume is not None` antes de usarlo; fallback determinista si falta).
- Devuelve `{"keypoints": [x, y], "tool_tissue_distance": float}` para
  Task 1 y `np.ndarray (3,3) float64` para Task 2 — sin cambios.
- Todo el cuerpo de `inference()` sigue envuelto en `try/except` con
  fallback numérico (regla de oro del archivo, sin tocar).

## Evaluación local

Vendor ya re-sincronizado con la corrección oficial (`GROUND_TRUTH_DISTANCE_SCALE
= 7.8125`, `MAX_THRESHOLD_DIST = 20`), así que `eval/run_local.py` puntúa con
las mismas constantes que usará Codabench ahora.

```powershell
python eval/run_local.py --submission submissions/r05-dist-rescaled --task keypoints --save
$env:KMP_DUPLICATE_LIB_OK = "TRUE"   # solo necesario para --task registration en Windows
python eval/run_local.py --submission submissions/r05-dist-rescaled --task both --save
```

Resultado real sobre los 5 casos del Mock Test (`local_score.json`):

| | keypoint_auc | distance_auc | final_score (Task 1) | Task 2 |
|---|---:|---:|---:|---:|
| r04 (constantes viejas) bajo vendor viejo | 0.854545 | 0.090909 | 0.625455 | 0.000000 |
| r04 (constantes viejas) bajo vendor **corregido** | 0.854545 | **0.000000** | 0.598182 | — |
| **r05 (constantes reescaladas) bajo vendor corregido** | 0.854545 | **0.133333** | **0.638182** | 0.000000 |

`keypoint_auc` no cambia entre r04 y r05 (ninguna de las dos constantes
tocadas afecta al keypoint) — es el control esperado. `distance_auc` sube de
`0.0` a `0.133333` solo por reescalar las dos constantes, sobre el mismo
vendor corregido y los mismos pesos. **Advertencia obligatoria** (ya
documentada en `RESULTS.md` para r01/r03): el harness local de 5 casos NO
predice el score de Codabench (r01 fue optimista local→real, r03 fue
pesimista) — este número confirma la dirección del efecto, no su magnitud.
Detalle y estimación cuantitativa en `experiments/86-t1-margen/INFORME.md`.

SHA-256:

- `inference.py`: `B6280B2370BE17BE000198B5EE185454DA1660ECD39A291495C46847CE51AB73`
- `model_0.pth`: `6B73FC21BC16BA1AE004E35E875298036C88EA6D777303ABCF63015CBAA773DA` (idéntico a r04)
- `model_1.pth`: `C365077A30533BBA5BEF9874A78A1F1FC11E89471939600F231AAE9C4310B01D` (idéntico a r04)
