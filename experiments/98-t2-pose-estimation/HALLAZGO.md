# T2-98 — Task 2 es estimación de pose, no registración de imágenes

**Fecha**: 2026-08-19. Hallazgo verificado sobre los 1214 casos de train.

## Qué se midió

Las anotaciones de Task 2 (`data/_annotations/Task 2/*/*.json`) contienen las
poses espaciales 3D de tres elementos: `Opmi.Spatial`, `iOCT Microscope.Spatial`
y `Eyeball.Spatial`. Desviación típica de cada campo sobre los 1214 casos:

| campo | sd por componente |
|---|---|
| `Opmi.Translation` | 0.0000 0.0000 0.0000 |
| `Opmi.Rotation` | 0.0000 0.0000 0.0000 0.0000 |
| `iOCT.Translation` | 0.0000 0.0000 0.0000 |
| **`iOCT.Rotation`** | **0.0594 0.0901 0.0539 0.2088** |
| `Eyeball.Translation` | 0.0000 0.0000 0.0000 |
| **`Eyeball.Rotation`** | **0.0265 0.0045 0.0082 0.6609** |
| matriz GT (6 params) | 52.70 52.15 117.85 52.03 52.64 142.24 |

**Solo dos cosas varían en todo el dataset**: la rotación del escáner iOCT y la
rotación del ojo. El microscopio es completamente fijo, y nada se traslada.

## La matriz GT es derivable de esas dos rotaciones

Regresión de los 6 parámetros reales de la matriz sobre los 8 componentes de
los dos cuaterniones (n=1214):

| modelo | R² por parámetro | error abs medio |
|---|---|---|
| lineal | 0.966 0.964 0.929 0.965 0.966 0.951 | 6.9 7.1 25.1 7.0 6.9 23.1 |
| **+ términos cuadráticos** | **0.976 0.975 0.994 0.975 0.976 0.994** | **5.7 5.7 6.7 5.7 5.7 7.0** |

Los términos cuadráticos son los que corresponden a la estructura real: una
matriz de rotación es cuadrática en el cuaternión. Con la composición
geométrica exacta el ajuste debería ser prácticamente perfecto; esta regresión
cruda ya deja el error de los términos de traslación en **5.7–7.0 px**, dentro
de la ventana de 0–10 px que puntúa la métrica.

## Por qué reordena el proyecto

Todas las vías intentadas hasta hoy —correspondencia de vasos, descriptores
comunes aprendidos, correlación de fase, instrumento como landmark— asumían que
Task 2 consiste en **alinear dos imágenes**. No lo es: el simulador genera la
transformación a partir de dos rotaciones y todo lo demás está fijo.

Esto explica de forma consistente cada resultado anómalo acumulado:

- La vasculatura no corresponde a nivel de píxel (IoU 0.015–0.07): cierto, e
  irrelevante — no hace falta emparejarla.
- La pose constante puntúa 0.0000: cierto, porque las rotaciones sí varían.
- Siete equipos puntúan hasta 0.475 sin que exista correspondencia de
  apariencia: porque no la están usando.

## Vía propuesta

Predecir las dos rotaciones desde (fundus, volumen) y **componer la matriz
geométricamente**, en vez de regresar la matriz o las esquinas directamente.

Ventajas sobre lo intentado:

1. El objetivo son parámetros de rotación suaves, no una matriz con
   discontinuidades ni un problema de correspondencia.
2. La rotación del **ojo** debería ser legible del fundus (posición del disco
   óptico, orientación del árbol vascular) — señal monomodal, sin cruce.
3. La rotación del **escáner** determina qué región de la retina captura el
   volumen.
4. **Supervisión adicional sin explotar**: la documentación oficial declara que
   Task 1 anota `eye pose` e `iOCT-to-fundus transform` en cada frame, y Task 1
   tiene 61,691 frames — unas 50× más ejemplos que los 1214 de Task 2.

## Pendiente de verificar antes de construir

- Derivar la composición geométrica EXACTA (no la regresión cuadrática) y
  confirmar que reproduce la matriz GT a precisión numérica.
- Confirmar que Task 1 anota efectivamente `eye pose` con el mismo convenio.
- Medir qué precisión angular se necesita para quedar bajo 10 px de error de
  esquinas: fija el listón de la estimación de pose.
