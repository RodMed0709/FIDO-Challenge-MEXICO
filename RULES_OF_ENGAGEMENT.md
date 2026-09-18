# Reglas de entrega — FIDO Challenge

> **La documentación de Codabench contradice al código de ingestión. El código manda.**
> Todo lo de aquí está verificado leyendo `vendor/fido/Codabench Bundle/`
> (commit `5a84d17`, 2026-08-08), no la página web.

Este documento existe porque cuatro de los once equipos del leaderboard de Task 2
están en `0.000`. Casi con certeza no es que sus modelos sean malos: es que
incumplieron el contrato de abajo.

---

## 1. Firma de `inference` — 4 argumentos, no 3

```python
def load_model(model_path):
    ...

def inference(task_id, oct_volume, opmi_image, model):
    ...
```

La página de Codabench muestra `inference(oct_data, fundus_image, model)`.
**Está mal.** El propio código de ingestión trae este comentario:

> `"inference.py must define inference(task_id, oct_volume, opmi_image, model). Note: The documentation may show a different signature; this one is correct."`

`task_id` es `0` para Task 1 (keypoints) y `1` para Task 2 (registración).

## 2. Nombre del archivo de pesos

| Task | Archivo requerido |
|---|---|
| Task 1 — keypoints | `model_0.pth` |
| Task 2 — registración | `model_1.pth` |

**No** `model.pth`. La ingestión valida `REQUIRED_FILES = {"inference.py", "model_0.pth"}`
y aborta la corrida entera si falta. Un solo carácter de diferencia = cero.

## 3. Estructura del zip — plano

```
mi_submission.zip
├── inference.py        ← en la raíz
├── model_0.pth         ← en la raíz
└── requirements.txt    ← opcional
```

Si zipeas la carpeta en vez de su contenido, la ingestión lo detecta y lo dice
explícitamente en el error, pero igual es cero. Desde *dentro* de la carpeta:

```powershell
Compress-Archive -Path .\* -DestinationPath mi_submission.zip -Force
```

## 4. `requirements.txt` — nunca `numpy` ni `torch`

Ya están instalados en el evaluador. Declararlos dispara una reinstalación que
puede romper el entorno o comerse el presupuesto de tiempo.

## 5. Formas y tipos reales de los datos

Lo que dice la doc, contra lo que entrega la ingestión:

| | Doc | Real (verificado en Mock Test) |
|---|---|---|
| `opmi_image` | 512×512×3 | **1024×1024×3** uint8 RGB |
| `oct_volume` Task 1 | (2, 256, 256) | **(2, 512, 512)** uint8 — 2 B-scans |
| `oct_volume` Task 2 | (256, 256, 256) | **(128, 512, 512)** uint8 — 128 slices |

La ingestión **no redimensiona nada**: apila los PNG tal cual con
`np.stack([...], axis=0)`. No asumas tamaños fijos — lee `.shape`.

## 6. `oct_volume` puede ser `None`

- **Task 1**: la ingestión fuerza `oct_volume=None` en el **10 %** de los casos,
  elegidos con semilla `RANDOM_SEED + 1 = 2027`. Es determinista, no aleatorio
  entre corridas.
- **Task 2**: no lo fuerza, pero `load_oct_volume` devuelve `None` si el
  directorio del volumen no existe.

Una excepción por `None` en un caso **aborta la corrida completa**, no solo ese
caso. Toda `inference()` debe manejar `oct_volume is None`.

## 7. Presupuesto de tiempo

- **20 s por caso.** Superarlo no aborta: ese caso se registra como timeout y
  entra al AUC con un error de `MAX_THRESHOLD + 1`, o sea falla todos los
  umbrales. Cuenta en el denominador igual.
- **100 casos** evaluados, elegidos con semilla `2026` — el mismo subconjunto
  siempre.
- **10 inferencias de warmup** antes de cronometrar. El arranque en frío de CUDA
  y cualquier compilación JIT no cuentan.
- Timeout global del contenedor: 1800 s en Competition, **600 s en Final Round**.
  Con 100 casos eso son 6 s/caso de promedio en la fase final. **El límite real
  en la ronda final es el global, no el de 20 s por caso.**

## 8. Formato de retorno

**Task 1** — diccionario:
```python
{
    "keypoints": [x, y],            # lista de 2 floats
    "tool_tissue_distance": z,      # float, en píxeles reales de B-scan
}
```
El ground truth de distancia está almacenado ×10 y el scoring lo divide entre 10
antes de comparar. **Predice en píxeles reales, no ×10.**

**Task 2** — array `(3, 3)` de `float64`, homografía. La fila de abajo es
`[0, 0, 1]` (es afín en la práctica).

## 9. Cómo se calcula el score

**Task 1**: `0.7 × keypoint_AUC + 0.3 × distance_AUC`
donde cada AUC es el promedio de `fracción de casos con error ≤ t` sobre
`t = 0, 1, …, 10` enteros.

**Task 2**: AUC del error de esquinas. El scoring proyecta las 4 esquinas del
cuadrado unitario `(0,0), (1,0), (1,1), (0,1)` por la matriz predicha y por la
real, y promedia la distancia L2 entre las 4 parejas.

Consecuencia geométrica: con `M = [[a,b,tx],[c,d,ty],[0,0,1]]`, la traslación
aparece en las 4 esquinas y cada columna lineal solo en 2. **Un error en
`(tx,ty)` cuesta el doble que el mismo error en escala o rotación.**

## 10. Límites de submission

| Fase | Por día | Total | Timeout |
|---|---|---|---|
| Competition (hasta 20 ago 2026) | 3 | 20 | 1800 s |
| Final Round (20 ago – 4 sep 2026) | 5 | 100 | 600 s |

Además, los términos dicen: *"Each team may submit only once to the final test
evaluation per task."*

---

## Regla dura

**Nada sube a Codabench sin haber pasado antes por `eval/run_local.py`.**

Ese script importa la ingestión y el scoring vendorizados sin modificarlos, así
que lo que reporta es lo que reportará Codabench salvo por los datos ocultos.
Con 20 submissions totales en Competition, quemar una en un `model.pth` mal
nombrado es inaceptable.

```bash
python eval/run_local.py --submission submissions/<id> --task registration --save
```
