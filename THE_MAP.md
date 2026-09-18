# El mapa — FIDO Challenge

Qué existe, dónde vive, en qué estado está. Documento vivo: se actualiza cuando
cambia la realidad, no al final.

**Última actualización**: 2026-08-18

---

## Reloj

| Hito | Fecha | Estado |
|---|---|---|
| Fase Practice | 1–30 jun 2026 | pasada |
| **Fase Competition** | 30 jun – **20 ago 2026** | **corriendo — 2 días** |
| **Final Round** | 20 ago – **4 sep 2026** | **la que decide premios** |

Premios: ~€350 por Task 1, ~€350 por Task 2, ~€350 al enfoque más innovador.
Más coautoría (hasta 2 personas por equipo) en el paper conjunto.

---

## Estructura del repo

```
FIDO_CHALLENGE/
├── README.md                punto de entrada ← empieza aquí
├── CONSTITUTION.md          reglas inviolables
├── RULES_OF_ENGAGEMENT.md   contrato exacto de submission ← LEER ANTES DE ENTREGAR
├── ATTACK_LADDER.md         peldaños, hipótesis, resultados
├── THE_MAP.md               este archivo
├── EXECUTION_MAP.md         tablero cerrado de Task 1 y Task 2
├── CLAUDE.md                instrucciones para agentes
├── .env                     credenciales reales (NO se commitea)
├── .env.example             plantilla
├── analysis/
│   ├── extract_annotations.py     JSONs sin descomprimir los 83 GB
│   ├── verify_task2_structure.py  la prueba de los 4 DOF
│   ├── explore_task1_gt.py        distribuciones del GT de Task 1
│   ├── render_task2_figures.py    figuras desde datos reales
│   └── build_task2_explainer.py   ensambla el explicador HTML
├── data/                    datos locales (gitignorado)
│   ├── Mock Test/           validación local — 349 MB, extraído
│   ├── Task 2/              10 zips, 27.3 GB
│   ├── Task 1/              10 zips, 55.6 GB
│   ├── _annotations/        solo los JSON, extraídos de los zips
│   └── _sample/             un caso completo para las figuras
├── docs/
│   ├── superpowers/specs/   el diseño de Task 2
│   ├── figures/             JPEGs del explicador
│   ├── task2_explainer_template.html
│   └── task2_explainer.html builds del explicador
├── eval/
│   └── run_local.py         harness byte-exacto ← única puerta a Codabench
├── infra/
│   ├── upload_to_runpod.py  local → volumen S3
│   ├── bootstrap_pod.sh     entorno del pod (torch cu128 + Jupyter)
│   └── start_jupyter.sh     JupyterLab en el 8888
├── vendor/
│   ├── fido/                repo oficial CONGELADO — no editar
│   └── fido_COMMIT.txt      5a84d17, 2026-08-08
└── submissions/
    └── r00-smoke/           prueba del contrato de entrega
```

## Explicadores publicados

- **"El cuadrado del iOCT"** — explicador visual de Task 2. La versión
  renderizada lleva imágenes del dataset y por eso **no se publica aquí**:
  el dataset es CC BY-NC-ND y no se redistribuye.

  Fuente en `docs/task2_explainer_template.html`. Para actualizarlo: editar la
  plantilla, correr `python analysis/build_task2_explainer.py`, y republicar
  **pasando esa misma URL** para no crear un artifact nuevo.

---

## Infraestructura

| Pieza | Valor | Estado |
|---|---|---|
| Volumen RunPod | `<VOLUME_ID>`, datacenter EU-RO-1 | activo |
| Endpoint S3 | `https://s3api-eu-ro-1.runpod.io` | verificado |
| GPU de entrenamiento | RTX 5090 — Blackwell, `sm_120`, 32 GB | — |
| PyTorch | **cu128 obligatorio** (`sm_120` no está en cu121/cu124) | en bootstrap |
| Jupyter | JupyterLab en el venv del volumen, kernel `FIDO (cu128)` | escrito |

**Trampa conocida**: RunPod pone Cloudflare enfrente de su gateway S3, que mata
peticiones de más de ~100 s con un `524`. El tiempo de cada parte multipart es
`chunk × streams ÷ ancho_de_banda`; con partes de 64 MB sobre un enlace lento
cada parte tardaba ~12 min y fallaba siempre. Configuración actual: partes de
8 MB, 4 streams, archivos secuenciales.

**Segunda trampa**: RunPod responde `403` en vez de `404` a `HeadObject` sobre
una llave inexistente. La detección de "ya existe" usa `ListObjectsV2`.

---

## Estado del dataset

| Conjunto | Local | En el volumen |
|---|---|---|
| Mock Test (349 MB) | ✅ extraído | ✅ `raw/Mock_Test.tar` |
| Task 2 (27.3 GB, 10 zips) | ✅ `data/Task 2/` | ✅ **verificado por checksum** |
| Task 1 (55.6 GB, 10 zips) | ✅ `data/Task 1/` | ✅ **verificado por checksum** |

**Dataset completo arriba: 83.00 GB de 170 GB (49 %).** El volumen se expandió de
90 a 170 GB el 2026-08-15. Sin partes huérfanas.

La transferencia sobrevivió a tres apagones, dos caídas de internet y un cuelgue
de sockets, sin perder un byte. Aprendizajes que quedaron en el código:

- Los sockets colgados tras un corte **no dan error**: esperan al `read_timeout`.
  Con 180 s el proceso parecía muerto durante minutos. Bajado a 60 s, más un
  vigilante que mata el proceso si pasan 150 s sin avanzar.
- Los procesos en background de Claude **se matan solos**. Para transferencias
  largas hay que lanzarlas desacopladas (`infra/upload_task1_loop.cmd`).
- RunPod a veces **consolida** un objeto multipart y le pone un MD5 del archivo
  completo en vez del hash compuesto. Comparar contra el compuesto da un falso
  "corrupto" — `verify_upload.py` ya distingue los dos casos por el sufijo `-N`.

**Herramientas de transferencia** (aprendizajes que costaron):

| Script | Cuándo usarlo |
|---|---|
| `infra/upload_to_runpod.py` | Subida normal. Reintenta **por archivo**: un corte al 90 % reempieza desde 0 |
| `infra/resumable_upload.py` | Conexión inestable. Reintenta **por parte**: un corte pierde 8 MB, no 4 GB. Probado con caída simulada |
| `infra/verify_upload.py` | Confirmar integridad. Recalcula el ETag multipart en local — detecta corrupción, no solo truncamiento |

### ✅ Resuelto — el volumen se expandió a 170 GB

`<VOLUME_ID>` ("FIDO_CHALLENGE") se expandió de 90 → **170 GB** el 2026-08-15
(decisión de Rodrigo). Ambas tasks subieron completas y se verificaron por
checksum: 27.26 GB (Task 2) + 55.62 GB (Task 1) + Mock Test = 83 GB usados,
49 % del volumen. Queda espacio de sobra para descomprimir, el venv, y
checkpoints de entrenamiento.

---

## Formato real de los datos

Verificado contra el Mock Test, **no** contra la documentación (que miente).

```
<Task N>/Scenario_NN/
├── Numerical/<frame>.json          anotaciones + ground truth
├── Stereo Left/<frame>/
│   ├── microscope.png              1024×1024 RGB   ← la imagen de fundus
│   ├── visibility.png
│   └── Segmentation/
│       ├── arteriesorveins.png     ← máscara de vasos, útil como supervisión
│       ├── cannula.png
│       ├── endoilluminator.png
│       ├── forceps.png
│       └── ilm.png
└── iOCT Microscope/
    ├── properties.json             512×512, 40 mm profundidad, patrón "Cross"
    ├── Bscan/<frame>/00.png,01.png     Task 1 — 2 B-scans de 512×512
    └── Volume/<frame>/*.png            Task 2 — 128 slices de 512×512
```

**Contenido de `Numerical/<frame>.json`**:
- `Opmi/Spatial`, `iOCT Microscope/Spatial`, `Eyeball/Spatial` — poses 3D
- `Surgical Tool/<TOOL>/Spatial` — puntas de instrumento en 3D
- `Keypoints/` — proyecciones 2D: puntas de instrumento, **crosshair del iOCT**,
  y ~55 landmarks de vasculatura (`V_0..V_32`, `A_0..A_22`)
- `Ground Truth/Task 1` — `[x, y, distancia×10]`
- `Ground Truth/Task 2` — afín 3×3

---

## Geometría de Task 2 — lo que ya sabemos

La matriz mapea coordenadas normalizadas `[-1,1]` con origen al centro →
píxeles del `microscope.png` de 1024×1024. **Es exactamente la geometría del
crosshair del iOCT** (verificado a 4 decimales).

Es una **similitud reflejada de 4 DOF**, verificado sobre los **1214** snapshots
de entrenamiento: determinante negativo en el 100 %, ortogonalidad dentro de 10°
en el 100 %, razón de normas dentro de 5 % en el 99.2 %.

Escala **160 ± 13.7 px** (rango 126–194) — *no* constante; una medición previa
sobre un solo escenario daba 217 ± 5 y era engañosa. Forzar la forma de 4 DOF
deja un residuo de 1.74 px de media, o sea un **techo de AUC en 0.794**.

El dataset trae **1214 snapshots de entrenamiento en Task 2**, no los 150 que
anuncia la documentación.

**El crosshair no está renderizado en la imagen.** La señal utilizable está en
la proyección en-face del volumen (`V.mean(axis=1)` con `V` de forma
`(128,512,512)`), donde se ve vasculatura retiniana nítida y la silueta del
instrumento.

Detalle completo en `ATTACK_LADDER.md`, peldaños R2 y R3.

---

## Tamaños reales del dataset

La documentación se queda corta en las dos tasks. Contado de los
`Numerical/*.json` de los zips:

| | Doc dice | Real |
|---|---|---|
| Task 2 — snapshots de entrenamiento | 150 | **1,214** |
| Task 1 — frames de entrenamiento | ~25,000 | **61,691** |

---

## Decisiones tomadas

| Fecha | Decisión | Razón |
|---|---|---|
| 2026-08-14 | Task 2 antes que Task 1 | campo más débil: 2° lugar en 0.296, cuatro equipos en 0.000 |
| 2026-08-14 | Vendorizar el repo oficial congelado | garantiza que el scoring local sea byte-exacto |
| 2026-08-14 | Descartar detección de crosshair | no está renderizado en la imagen |
| 2026-08-14 | Parametrizar con 4 DOF, no 6 | la estructura del GT lo justifica y evita matrices degeneradas |
| 2026-08-14 | Subida secuencial con partes de 8 MB | el paralelismo no daba throughput y provocaba `524` |

---

## Presupuesto de tiempo — medido

La ingestión carga el volumen **fuera** de la ventana cronometrada por caso,
pero **dentro** del timeout global del contenedor.

| Carga del volumen (128 PNG de 512×512) | por caso | 100 casos |
|---|---|---|
| completo — lo que hace la ingestión | 459 ms | **46 s** |

**Final Round**: 600 s globales − 46 s de I/O ≈ **5.5 s por caso** de inferencia
real. El límite nominal de 20 s por caso es irrelevante ahí; manda el global.

---

## Abierto

- [x] ~~Aprobación de diseños y planes de Task 1/Task 2~~ — cerrados el 2026-08-18
      en `EXECUTION_MAP.md` y `docs/superpowers/{specs,plans}/`.
- [ ] Ejecutar T1-80 y T2-80 según el mapa maestro; no abrir experimentos fuera
      del árbol de decisión cerrado.
- [x] ~~Corrida de humo de `eval/run_local.py`~~ — verde, ambas tasks
- [x] ~~Medir el costo de cargar 128 slices PNG~~ — 46 s por 100 casos
- [ ] Arrancar un pod y correr `infra/bootstrap_pod.sh` de verdad
- [ ] Reverificar la estructura de 4 DOF sobre los 150 snapshots de
      entrenamiento (hasta ahora solo medida sobre 5 frames de un escenario)
- [ ] Confirmar el segundo coautor del equipo
- [ ] ¿Inicializar git? El repo aún no está bajo control de versiones. No lo
      hice porque commitear no estaba pedido explícitamente.
