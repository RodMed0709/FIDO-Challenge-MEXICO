# 20 — Task 2: ¿sirve el patrón de vasos como puente entre OCT y fundus?

Peldaño de la escalera: **T2-R4, Motor 1** (`ATTACK_LADDER.md`).

## Escalera

| run | qué es | métrica |
|---|---|---|
| `../30-t2-baseline-roto/` | control: baseline sin vasos | `corner_error` ~165-200 px |
| **20a** | template matching clásico (NCC) sobre máscaras de vaso | **`corner_error` 573.69 px, `corner_auc` 0.0000** |
| **20b** | mismo motor con la señal de vasos transformada (binarizada, dilatada 2-21px) | oráculo NCC 0.007-0.018 |

## Expectativa pre-registrada

+0.05 a +0.15 de AUC sobre el baseline. Fundada en literatura con buen
precedente empírico: RetinaMatch >94% de éxito, Noyel 96% sobre 271 pares, ambos
usando estructura vascular. Marcada `[abs]` (solo abstracts leídos).

## 🔴 RESULTADO (2026-08-17) — REFUTADO, y no por un bug

**El método no falla por implementación: la señal misma no existe.**

Esto se estableció con dos pruebas independientes, no con conjeturas:

1. **Caso sintético autoconsistente**: se genera un patrón conocido, se le aplica
   una transformación conocida, y se pide al matcher que la recupere.
   Resultado: `corner_error = 0.09 px`, `match_score = 0.9965`.
   → La cadena `_build_patch → matchTemplate → fit_closed_form_similarity` es
   matemáticamente correcta. **No hay bug de geometría.**

2. **Oráculo con el GT real**: se toman los parámetros REALES del ground truth
   (sin ninguna búsqueda) y se mide la correlación entre el patch del en-face
   alineado en la posición correcta y la región real del fundus:

   | caso | oracle NCC |
   |---|---|
   | 00031 | -0.0072 |
   | 00066 | 0.0462 |
   | 00099 | -0.0116 |
   | 00177 | 0.0001 |

   **Ruido puro, en la respuesta perfecta.** Ningún algoritmo de búsqueda puede
   mejorar eso, porque no hay nada que encontrar.

**Segunda ronda (20b)**: se probó si otra representación de la señal la rescata
— binarizar, MIP, dilatación morfológica con kernels de 2 a 21 px, y búsqueda
con traslación libre. Mejor oráculo NCC alcanzado: **0.018** (umbral orientativo
pre-registrado: 0.3). La búsqueda libre sí llega a NCC 0.24-0.31 pero **en una
posición que no coincide con el GT en 4 de 4 casos** — falso positivo por
autosimilitud del árbol vascular (las curvas finas se parecen entre sí en
cualquier parte de la imagen). Ese es exactamente el modo de fallo detrás de los
573 px.

**Causa raíz**: `enface_vessel_density` (fracción de profundidad ocupada por la
clase `ArteriesOrVeins`) es demasiado dispersa — ~4-6% de píxeles no-cero, valor
máximo 1/512 — y no reproduce la topología continua del árbol vascular que sí se
ve en el fundus.

**Costo**: $0 (todo CPU local, contra el Mock Test).

## Hallazgo secundario, sin cerrar

La escala del GT en el Mock Test (**207-224**) cae **fuera** del rango medido en
los 1214 casos de entrenamiento (**125.9-196.9**, media 159.6). Rangos disjuntos.
Si el test oculto se parece al Mock Test, cualquier prior de escala calibrado en
train está sesgado. **No hardcodear priors de escala hasta resolver esto.**

## Qué NO volver a intentar

- Template matching clásico sobre `enface_vessel_density` tal como está definida,
  con cualquier transformación morfológica simple. Cerrado con evidencia oráculo.
- Traducir OCT→fundus con GAN antes de matchear (ya descartado por la literatura).

## Qué sí queda abierto

Motor 2 (registro aprendido sobre probabilidades de vaso) o combinar la señal de
vasos con la silueta del instrumento, que también es visible en el en-face y
sigue sin explotarse.

## Scripts

- `analysis/verify_vessel_template_match_synthetic.py` — caso sintético + oráculo
- `analysis/verify_vessel_signal_candidates.py` — las candidatas de señal A-F
