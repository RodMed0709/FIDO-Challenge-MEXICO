# Verificación de la convención en-face de Task 2

Root: `data\Mock Test\Task 2`. Casos solicitados: n=5; casos válidos con segmentación: n=5; omitidos: n=0. Seed=0; null=200 muestras uniformes por keypoint.

Los radios 0.005, 0.01, 0.02 y 0.04 son fracciones del lado del cuadrado unitario. La transformada de distancia usa el espaciado físico de cada eje de la rejilla 128 x 512.

## Interpretación pre-registrada

- Si una variante da ratio observado/null >= 1.5 consistentemente en radios chicos y las otras quedan en ~1.0, esa es la convención correcta y el arnés estaba roto.
- Si las ocho quedan en ~1.0, la convención no es el problema y la refutación de T2-R11 es sólida.
- Si todas suben parejo al corregir la anisotropía, el bug era el radio, no el eje.

## C — Instrumento

Casos con >=1 keypoint de instrumento dentro de la huella: 4/5 (0.8000; n=5).

## C — Tasas por variante

Cada celda de radio es `observado / null / ratio`; las tasas son pooled por keypoint.

| variante | n puntos | n null | r=0.005 | r=0.01 | r=0.02 | r=0.04 |
|---|---:|---:|---:|---:|---:|---:|
| identity__keep_u__keep_v | 8 | 1600 | 0.0000 / 0.0063 / 0.0000 | 0.0000 / 0.0094 / 0.0000 | 0.0000 / 0.0119 / 0.0000 | 0.0000 / 0.0231 / 0.0000 |
| identity__keep_u__flip_v | 8 | 1600 | 0.0000 / 0.0044 / 0.0000 | 0.0000 / 0.0069 / 0.0000 | 0.0000 / 0.0125 / 0.0000 | 0.0000 / 0.0238 / 0.0000 |
| identity__flip_u__keep_v | 8 | 1600 | 0.0000 / 0.0069 / 0.0000 | 0.0000 / 0.0081 / 0.0000 | 0.0000 / 0.0138 / 0.0000 | 0.0000 / 0.0281 / 0.0000 |
| identity__flip_u__flip_v | 8 | 1600 | 0.0000 / 0.0056 / 0.0000 | 0.0000 / 0.0100 / 0.0000 | 0.0000 / 0.0119 / 0.0000 | 0.0000 / 0.0213 / 0.0000 |
| transpose__keep_u__keep_v | 8 | 1600 | 0.0000 / 0.0056 / 0.0000 | 0.0000 / 0.0063 / 0.0000 | 0.0000 / 0.0106 / 0.0000 | 0.0000 / 0.0169 / 0.0000 |
| transpose__keep_u__flip_v | 8 | 1600 | 0.0000 / 0.0044 / 0.0000 | 0.0000 / 0.0050 / 0.0000 | 0.0000 / 0.0088 / 0.0000 | 0.0000 / 0.0187 / 0.0000 |
| transpose__flip_u__keep_v | 8 | 1600 | 0.0000 / 0.0075 / 0.0000 | 0.0000 / 0.0100 / 0.0000 | 0.0000 / 0.0156 / 0.0000 | 0.0000 / 0.0275 / 0.0000 |
| transpose__flip_u__flip_v | 8 | 1600 | 0.0000 / 0.0031 / 0.0000 | 0.0000 / 0.0069 / 0.0000 | 0.0000 / 0.0125 / 0.0000 | 0.0000 / 0.0200 / 0.0000 |

## A — Vasculatura

Cada celda de radio es `observado / null / ratio`; las tasas son pooled por keypoint.

| variante | n puntos | n null | r=0.005 | r=0.01 | r=0.02 | r=0.04 |
|---|---:|---:|---:|---:|---:|---:|
| identity__keep_u__keep_v | 4 | 800 | 0.2500 / 0.1175 / 2.1277 | 0.2500 / 0.1938 / 1.2903 | 0.5000 / 0.2913 / 1.7167 | 0.5000 / 0.4675 / 1.0695 |
| identity__keep_u__flip_v | 4 | 800 | 0.0000 / 0.1087 / 0.0000 | 0.0000 / 0.1938 / 0.0000 | 0.0000 / 0.2825 / 0.0000 | 0.0000 / 0.4550 / 0.0000 |
| identity__flip_u__keep_v | 4 | 800 | 0.5000 / 0.1000 / 5.0000 | 0.5000 / 0.1650 / 3.0303 | 0.5000 / 0.2725 / 1.8349 | 0.5000 / 0.4537 / 1.1019 |
| identity__flip_u__flip_v | 4 | 800 | 0.2500 / 0.0862 / 2.8986 | 0.2500 / 0.1650 / 1.5152 | 0.5000 / 0.2550 / 1.9608 | 0.7500 / 0.4412 / 1.6997 |
| transpose__keep_u__keep_v | 4 | 800 | 0.0000 / 0.1025 / 0.0000 | 0.0000 / 0.1675 / 0.0000 | 0.0000 / 0.2612 / 0.0000 | 0.2500 / 0.4700 / 0.5319 |
| transpose__keep_u__flip_v | 4 | 800 | 0.5000 / 0.0838 / 5.9701 | 0.5000 / 0.1562 / 3.2000 | 0.5000 / 0.2412 / 2.0725 | 0.5000 / 0.4275 / 1.1696 |
| transpose__flip_u__keep_v | 4 | 800 | 0.2500 / 0.0975 / 2.5641 | 0.2500 / 0.1625 / 1.5385 | 0.2500 / 0.2787 / 0.8969 | 0.5000 / 0.4525 / 1.1050 |
| transpose__flip_u__flip_v | 4 | 800 | 0.0000 / 0.0963 / 0.0000 | 0.0000 / 0.1600 / 0.0000 | 0.0000 / 0.2550 / 0.0000 | 0.2500 / 0.4325 / 0.5780 |

## Veredicto

**not_convention** — Ninguna variante muestra enriquecimiento >= 1.5 consistente en los dos radios chicos; la convención no explica el resultado y la refutación de T2-R11 se sostiene.
