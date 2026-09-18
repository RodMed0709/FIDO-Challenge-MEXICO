# Verificación de la convención en-face de Task 2

Root: `/workspace/data/Task2`. Casos solicitados: n=1214; casos válidos con segmentación: n=1214; omitidos: n=0. Seed=0; null=200 muestras uniformes por keypoint.

Los radios 0.005, 0.01, 0.02 y 0.04 son fracciones del lado del cuadrado unitario. La transformada de distancia usa el espaciado físico de cada eje de la rejilla 128 x 512.

## Interpretación pre-registrada

- Si una variante da ratio observado/null >= 1.5 consistentemente en radios chicos y las otras quedan en ~1.0, esa es la convención correcta y el arnés estaba roto.
- Si las ocho quedan en ~1.0, la convención no es el problema y la refutación de T2-R11 es sólida.
- Si todas suben parejo al corregir la anisotropía, el bug era el radio, no el eje.

## C — Instrumento

Casos con >=1 keypoint de instrumento dentro de la huella: 279/1214 (0.2298; n=1214).

## C — Tasas por variante

Cada celda de radio es `observado / null / ratio`; las tasas son pooled por keypoint.

| variante | n puntos | n null | r=0.005 | r=0.01 | r=0.02 | r=0.04 |
|---|---:|---:|---:|---:|---:|---:|
| identity__keep_u__keep_v | 469 | 93800 | 0.0085 / 0.0073 / 1.1747 | 0.0128 / 0.0111 / 1.1561 | 0.0128 / 0.0171 / 0.7491 | 0.0213 / 0.0305 / 0.6995 |
| identity__keep_u__flip_v | 469 | 93800 | 0.0043 / 0.0066 / 0.6441 | 0.0064 / 0.0100 / 0.6403 | 0.0064 / 0.0160 / 0.4008 | 0.0107 / 0.0300 / 0.3559 |
| identity__flip_u__keep_v | 469 | 93800 | 0.0085 / 0.0066 / 1.2862 | 0.0085 / 0.0104 / 0.8222 | 0.0128 / 0.0160 / 0.8016 | 0.0235 / 0.0292 / 0.8044 |
| identity__flip_u__flip_v | 469 | 93800 | 0.0213 / 0.0069 / 3.0817 | 0.0235 / 0.0105 / 2.2426 | 0.0256 / 0.0163 / 1.5717 | 0.0426 / 0.0296 / 1.4409 |
| transpose__keep_u__keep_v | 469 | 93800 | 0.0021 / 0.0071 / 0.2999 | 0.0085 / 0.0109 / 0.7828 | 0.0085 / 0.0169 / 0.5051 | 0.0128 / 0.0302 / 0.4239 |
| transpose__keep_u__flip_v | 469 | 93800 | 0.0107 / 0.0066 / 1.6051 | 0.0128 / 0.0104 / 1.2295 | 0.0171 / 0.0162 / 1.0554 | 0.0277 / 0.0298 / 0.9292 |
| transpose__flip_u__keep_v | 469 | 93800 | 0.0021 / 0.0068 / 0.3125 | 0.0021 / 0.0107 / 0.1988 | 0.0043 / 0.0165 / 0.2582 | 0.0192 / 0.0303 / 0.6338 |
| transpose__flip_u__flip_v | 469 | 93800 | 0.0256 / 0.0070 / 3.6364 | 0.0362 / 0.0106 / 3.4309 | 0.0682 / 0.0163 / 4.1775 | 0.0959 / 0.0304 / 3.1568 |

## A — Vasculatura

Cada celda de radio es `observado / null / ratio`; las tasas son pooled por keypoint.

| variante | n puntos | n null | r=0.005 | r=0.01 | r=0.02 | r=0.04 |
|---|---:|---:|---:|---:|---:|---:|
| identity__keep_u__keep_v | 1144 | 228800 | 0.0848 / 0.0853 / 0.9937 | 0.1521 / 0.1611 / 0.9443 | 0.2622 / 0.2620 / 1.0011 | 0.4545 / 0.4541 / 1.0011 |
| identity__keep_u__flip_v | 1144 | 228800 | 0.0787 / 0.0855 / 0.9202 | 0.1530 / 0.1629 / 0.9393 | 0.2526 / 0.2639 / 0.9572 | 0.4642 / 0.4540 / 1.0224 |
| identity__flip_u__keep_v | 1144 | 228800 | 0.0892 / 0.0846 / 1.0543 | 0.1556 / 0.1616 / 0.9631 | 0.2483 / 0.2618 / 0.9484 | 0.4388 / 0.4537 / 0.9672 |
| identity__flip_u__flip_v | 1144 | 228800 | 0.0726 / 0.0849 / 0.8549 | 0.1600 / 0.1601 / 0.9990 | 0.2570 / 0.2610 / 0.9845 | 0.4528 / 0.4523 / 1.0011 |
| transpose__keep_u__keep_v | 1144 | 228800 | 0.0726 / 0.0846 / 0.8575 | 0.1451 / 0.1608 / 0.9026 | 0.2666 / 0.2625 / 1.0157 | 0.4432 / 0.4557 / 0.9726 |
| transpose__keep_u__flip_v | 1144 | 228800 | 0.0787 / 0.0850 / 0.9259 | 0.1503 / 0.1615 / 0.9307 | 0.2456 / 0.2622 / 0.9368 | 0.4484 / 0.4529 / 0.9902 |
| transpose__flip_u__keep_v | 1144 | 228800 | 0.0892 / 0.0843 / 1.0575 | 0.1652 / 0.1609 / 1.0266 | 0.2710 / 0.2622 / 1.0334 | 0.4537 / 0.4536 / 1.0001 |
| transpose__flip_u__flip_v | 1144 | 228800 | 0.0795 / 0.0842 / 0.9444 | 0.1442 / 0.1593 / 0.9055 | 0.2483 / 0.2612 / 0.9503 | 0.4484 / 0.4532 / 0.9895 |

## Veredicto

**not_convention** — Ninguna variante muestra enriquecimiento >= 1.5 consistente en los dos radios chicos; la convención no explica el resultado y la refutación de T2-R11 se sostiene.
