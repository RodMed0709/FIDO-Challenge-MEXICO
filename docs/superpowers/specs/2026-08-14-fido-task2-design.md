# Diseño — FIDO Challenge Task 2 (registración iOCT → fundus)

**Fecha**: 2026-08-14
**Estado**: ✅ APROBADO (2026-08-15, verbal, sesión de voz). Aprobación explícita
de ambas tareas — ver `ATTACK_LADDER.md` sección Task 1 para el spec paralelo.
**Alcance**: Task 2. Task 1 tiene su propio bloque de peldaños en
`ATTACK_LADDER.md` (no un spec separado — la escalera hace ese papel).

---

## 1. El problema, después de mirar los datos

Codabench lo presenta como *"estimar la transformación rígida 3D→2D que alinea
el escaneo volumétrico de iOCT con la imagen de microscopio de fundus"*. Tras
analizar el ground truth real, es algo más específico y más tratable.

### 1.1 Qué es realmente la matriz

`Numerical/<frame>.json → ["Ground Truth"]["Task 2"]` es una afín 3×3 que mapea
coordenadas normalizadas `[-1,1]` con origen al centro → píxeles de la imagen
de microscopio de 1024×1024.

Es **exactamente la geometría del crosshair del iOCT**. Proyectando los puntos
medios de los bordes del cuadrado unitario por `M` se reproducen las
anotaciones `Keypoints/iOCT Microscope Crosshair` a cuatro decimales:

```
M @ [ 0, 1, 1] = (261.176, 361.250)  ==  Start 0 = [261.1756, 361.2502]
M @ [-1, 0, 1] = (235.517,  61.126)  ==  Start 1 = [235.5175,  61.1264]
M @ [ 1, 0, 1] = (-44.313, 384.306)  ==  End 1   = [-44.31268, 384.3064]
```

La traslación `(tx, ty)` es el punto medio de `Start 1`–`End 1`.

### 1.2 Solo tiene 4 grados de libertad

Verificado sobre **los 1214 snapshots de entrenamiento** de los 10 escenarios
(`analysis/verify_task2_structure.py`), no solo sobre el Mock Test:

| Prueba | Resultado |
|---|---|
| Determinante negativo | **100.00 %** |
| Razón `\|col0\|/\|col1\|` dentro de ±5 % | 99.18 % |
| Ángulo entre columnas a ±5° de 90° | 99.67 % |
| Ángulo entre columnas a ±10° de 90° | **100.00 %** |

Es una **similitud reflejada**:

```
M(tx, ty, θ, s) = [[ s·cosθ,  s·sinθ,  tx],
                   [ s·sinθ, -s·cosθ,  ty],
                   [      0,       0,   1]]
```

Distribución de los parámetros:

| parámetro | media | sd | rango |
|---|---|---|---|
| escala `s` | 160.4 | 13.7 | 126 – 194 |
| `tx` | 471.9 | 125.0 | 72 – 883 |
| `ty` | 467.0 | 139.9 | 102 – 793 |
| `θ` | — | — | −180° – 180° (círculo completo) |

**Corrección de una medición previa.** Sobre el único escenario del Mock Test la
escala salía 217 ± 5, lo que sugería que se podía asumir constante. A nivel de
población es 160 ± 13.7. Asumir la media daría errores de hasta 34 px contra un
presupuesto de 12: **la escala se predice, no se asume.**

Que θ cubra el círculo completo obliga a parametrizarla como `(cos θ, sin θ)`
normalizado. Un ángulo crudo tendría una discontinuidad justo en una zona
poblada.

### 1.2b Cuánto cuesta forzar los 4 DOF

Ajustando a cada matriz real su similitud reflejada más cercana y midiendo el
error de esquinas resultante con la métrica oficial:

| | |
|---|---|
| error medio | 1.74 px (sd 1.47, máx 9.25) |
| error < 2 px | 68.5 % |
| error < 5 px | 95.8 % |
| **techo de AUC con 4 DOF perfectos** | **0.7938** |

O sea: la restricción **no es gratis**. Un modelo perfecto restringido a 4 DOF
tope en 0.794. Sigue muy por encima del líder actual (0.475), así que la
parametrización se justifica — pero para pasar de ~0.79 hay que modelar el
residuo.

**Diseño resultante**: predecir la similitud de 4 DOF **más dos términos de
residuo** sobre la parte lineal. El modelo arranca en la variedad correcta (buen
condicionamiento, sin matrices degeneradas) pero no queda encerrado en ella.

### 1.3 Presupuesto de error derivado de la métrica

El scoring proyecta las 4 esquinas del cuadrado unitario `(0,0), (1,0), (1,1),
(0,1)` y promedia la distancia L2. Con `M = [[a,b,tx],[c,d,ty],[0,0,1]]`:

| esquina | proyecta a | qué error la afecta |
|---|---|---|
| (0,0) | (tx, ty) | solo traslación |
| (1,0) | (a+tx, c+ty) | traslación + col0 |
| (1,1) | (a+b+tx, c+d+ty) | traslación + col0 + col1 |
| (0,1) | (b+tx, d+ty) | traslación + col1 |

La traslación entra en las **4**; cada columna lineal solo en **2**. Entonces,
para un error medio de esquinas por debajo del umbral de 10 px:

- `|Δt| < 10 px`
- `Δs < ~12 px` (≈ 5 % relativo sobre s≈217)
- `Δθ < ~3°`

**La traslación vale el doble que la escala o el ángulo.** El diseño debe
gastar su capacidad ahí.

### 1.4 Lo que NO funciona

- **Detectar el crosshair visualmente**: no está renderizado en
  `microscope.png` ni tiene máscara en `Segmentation/`. Además suele salirse
  parcialmente del cuadro (coordenadas negativas). Descartado.
- **Predictor constante**: AUC medido = **0.0**. La matriz varía demasiado.

### 1.5 Dónde está la señal

La proyección en-face del volumen — promedio sobre el eje de profundidad,
`V.mean(axis=1)` con `V` de forma `(128, 512, 512)` — muestra **vasculatura
retiniana nítida** y la silueta del instrumento. Esa vasculatura también está en
el fundus. Ese es el par que hay que alinear.

El dataset además regala supervisión: máscaras de vasos
(`Segmentation/arteriesorveins.png`) y ~55 landmarks de vasculatura por frame
(`Keypoints/Vasculature/V_0..V_32`, `A_0..A_22`).

---

## 2. Arquitectura

### 2.1 Entradas

| Entrada | Forma | Preproceso |
|---|---|---|
| Fundus | 1024×1024×3 uint8 | → 512×512, normalizado |
| En-face OCT | de `(128,512,512)` → `(128,512)` | reescalado a 512×512 cuadrado |

El en-face se calcula como el promedio sobre el eje de profundidad. Los 128
slices cubren la misma extensión física que los 512 A-scans, así que hay que
reescalar el eje lento — no es un simple `resize` isotrópico.

`oct_volume` puede llegar `None`. En ese caso el canal de en-face se rellena con
ceros y una bandera se lo dice al modelo. El modelo se entrena con dropout de
esa modalidad para que ese camino no sea territorio inexplorado.

### 2.2 Modelo

Encoder de dos ramas con pesos no compartidos (las modalidades no comparten
estadísticas), fusión por correlación cruzada, y dos cabezas:

```
fundus 512² ──► encoder_f (timm, preentrenado) ──┐
                                                  ├─► correlación ─► cabeza de pose ─► (tx, ty, θ, s)
en-face 512² ─► encoder_o (mismo backbone)     ──┘
     │                                                └─► cabeza de vasos ─► máscara (auxiliar)
     └────────────────────────────────────────────────► cabeza de vasos ─► máscara (auxiliar)
```

**La cabeza de pose no predice 6 números sueltos.** Predice
`(tx, ty, cos θ, sin θ, log s, r0, r1)` — la similitud de 4 DOF más dos términos
de residuo sobre la parte lineal — y `M` se arma en forma cerrada. Razones:

1. No puede producir matrices degeneradas ni con determinante positivo. Sospecho
   que ahí está parte del origen de los cuatro `0.000` del leaderboard.
2. Coincide con la estructura real del GT, así que la capacidad del modelo se
   gasta en lo que varía en vez de en reaprender una restricción.
3. `θ` se predice como `(cos θ, sin θ)` normalizado, no como ángulo — evita la
   discontinuidad en ±180°, que importa porque los valores observados rondan
   128°–172°.

### 2.3 Loss

**La métrica exacta, no un sustituto.**

```python
def corner_loss(pred_params, gt_matrix):
    M_pred = compose(pred_params)            # (tx,ty,θ,s) -> 3x3
    corners = [(0,0), (1,0), (1,1), (0,1)]   # el cuadrado unitario del scoring
    return mean(||project(M_pred, c) - project(gt_matrix, c)|| for c in corners)
```

Es diferenciable y es literalmente lo que Codabench calcula. No hay brecha entre
lo que se optimiza y lo que se puntúa.

Loss total:
```
L = L_esquinas  +  λ_v · L_vasos(dice, ambas ramas)  +  λ_t · L_traslación
```

`L_traslación` es un término explícito sobre `(tx, ty)` porque pesa el doble en
la métrica y conviene que el gradiente lo refleje desde el inicio.

### 2.4 Refinamiento geométrico (peldaño R6)

La red da una estimación gruesa. Para bajar de "cerca" a `<10 px` sobre una
escala de 217 px (5 % de precisión), se refina:

1. Deformar la máscara de vasos del en-face con la `M` predicha
2. Búsqueda local sobre `(tx, ty, θ, s)` maximizando correlación con la máscara
   de vasos del fundus
3. Rango acotado a lo que la red ya garantiza; sin búsqueda global

**Restricción dura**: en Final Round son **600 s para 100 casos = 6 s por
caso**, incluyendo cargar 128 PNG. El refinamiento cabe o no cabe; se mide antes
de comprometerse.

---

## 3. Datos y entrenamiento

| | |
|---|---|
| Entrenamiento | **1214 snapshots** en 10 escenarios (65–183 por escenario) |
| Validación | Mock Test — `Task 2/Scenario_12`, 5 casos |
| Test | oculto, 30 snapshots |

**La documentación anuncia 150 snapshots de entrenamiento. Son 1214.** Contados
directamente de los `Numerical/*.json` de los diez zips. Ocho veces más datos de
lo planeado, lo que rebaja bastante el riesgo de sobreajuste.

Aun así, 1214 vistas de solo 10 ojos sintéticos no son 1214 muestras
independientes: los escenarios comparten anatomía. Mitigaciones:

- Validación cruzada por escenario (dejar un escenario fuera), no por frame —
  los frames del mismo escenario comparten anatomía y filtrarían.
- Aumentación que respete la geometría: rotar/trasladar el fundus y **componer
  la transformación correspondiente en el GT**, no solo aumentar la imagen.
  Esto multiplica los pares efectivos sin inventar anatomía.
- Encoders preentrenados congelados al inicio, descongelado gradual.

**El Mock Test no se toca para seleccionar hiperparámetros** más allá del score
agregado (`CONSTITUTION.md` §3). La selección se hace con la validación cruzada
por escenario.

---

## 4. Riesgos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Sobreajuste: 1214 vistas de solo 10 ojos | media | CV por escenario, aumentación geométrica, encoders congelados |
| El refinamiento no cabe en 5.5 s/caso | media | medido: la I/O se come 46 s de los 600 s globales |
| ~~Cargar 128 PNG domina el tiempo~~ | — | **medido**: 459 ms/caso, 46 s por 100 casos. Manejable |
| ~~La estructura de 4 DOF no se sostiene~~ | — | **verificado** sobre los 1214: det negativo 100 %, ortogonalidad 100 % |
| El techo de 0.794 de los 4 DOF limita | media | por eso se añaden 2 términos de residuo a la parametrización |
| El volumen de RunPod (90 GB) no alcanza | **alta** | decisión pendiente: expandir, submuestrear Task 1, o posponerla |
| Se acaban las submissions de Competition | baja | 20 totales; el harness local es la defensa |

Los dos riesgos que estaban marcados como principales quedaron cerrados con
medición. El que queda abierto y bloquea es el espacio en el volumen.

---

## 5. Qué NO está en alcance

- Task 1 — spec aparte, después
- Registración 3D real (estimación de pose volumétrica). El GT es una afín 2D
  de 4 DOF; tratarlo como problema 3D es resolver algo más difícil que lo que se
  puntúa.
- Cualquier cosa que no quepa en 6 s por caso.

---

## 6. Criterio de éxito

| Nivel | Score Task 2 | Qué significa |
|---|---|---|
| Piso | > 0.00 | el contrato de entrega funciona |
| Aceptable | 0.30 | top-3 con el leaderboard actual |
| Objetivo | 0.50 | pasa al líder actual (0.475) |
| Ambicioso | 0.70 | margen cómodo |
| Techo de 4 DOF puros | **0.794** | medido; pasarlo exige los términos de residuo |

---

## 7. Pendiente de decisión

1. **Aprobación de este diseño** — bloquea todo el código de modelo.
2. Backbone concreto (ConvNeXt-T, ResNet-50, o un ViT pequeño). Se decide con un
   barrido rápido en R4, no por gusto.
3. Si el refinamiento de R6 no cabe en tiempo, ¿se sacrifica precisión o se
   busca una versión más barata? Se decide con la medición en mano.
