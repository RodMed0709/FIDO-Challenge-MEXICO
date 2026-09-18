"""Submission r01 — primer modelo real.

Task 1: keypoint de la punta de la cánula (CNN + heatmap, soft-argmax sub-píxel)
        + distancia herramienta-tejido por segmentación de B-scan y medición
        geométrica.
Task 2: registración iOCT->fundus por correlación cruzada de features.

AUTOCONTENIDO A PROPÓSITO. El evaluador de Codabench no tiene instalado el
paquete `fido` de este repo, así que las definiciones de los modelos y el
preproceso se copian aquí íntegros. Cualquier cambio en `src/fido/models/` debe
replicarse en este archivo o los pesos dejarán de casar.

CONTRATO (verificado en vendor/fido/Codabench Bundle/, no en la doc web):
  - firma de 4 argumentos: inference(task_id, oct_volume, opmi_image, model)
  - task_id 0 = Task 1 (keypoints), 1 = Task 2 (registración)
  - pesos: model_0.pth (Task 1) / model_1.pth (Task 2)
  - opmi_image: (H, W, 3) uint8 RGB, típicamente 1024x1024 — NO redimensionado
    por la ingestión, hay que leer .shape
  - oct_volume Task 1: (2, 512, 512) uint8 | Task 2: (128, 512, 512) uint8
  - oct_volume puede ser None (10% de Task 1, semilla determinista 2027)
  - Task 1 devuelve {"keypoints": [x, y], "tool_tissue_distance": float}
    con la distancia en PÍXELES REALES (el GT está x10 y el scoring divide)
  - Task 2 devuelve np.ndarray (3, 3) float64

REGLA DE ORO DE ESTE ARCHIVO: una excepción en un caso aborta la corrida
COMPLETA, no solo ese caso. Todo camino de inferencia está envuelto y tiene un
fallback numérico. Es preferible un caso malo a cien casos perdidos.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Constantes medidas sobre los datos reales de entrenamiento. No son magia:
# cada una tiene su procedencia anotada.
# --------------------------------------------------------------------------

# Ajuste lineal distancia<-gap de píxeles, R^2=0.9916 sobre 92,012 mediciones
# (analysis/verify_task1_distance_geometry.py, peldaño T1-R2). Ajustado
# originalmente contra GT/10 (GROUND_TRUTH_DISTANCE_SCALE viejo). El 2026-08-19
# los organizadores corrigieron esa constante a GT/7.8125 = 1.28x el target
# viejo (ver vendor/fido/.../scoring_keypoints.py y NOW.md). Como el ajuste
# es OLS y el target completo se reescala por un factor constante k=1.28, la
# proyección lineal correcta es exactamente k*a_viejo, k*b_viejo (no un nuevo
# ajuste: es álgebra exacta de mínimos cuadrados bajo reescalado lineal del
# target, R^2 no cambia). No se reentrenó nada.
DISTANCE_SCALE_A = 0.9370  # 0.7320 * 1.28
DISTANCE_SCALE_B = 3.4277  # 2.6779 * 1.28

# Mejor constante posible para la distancia cuando no hay OCT o la segmentación
# no produce una medición: daba distance_auc=0.0445 sobre los 61,691 casos bajo
# la escala/umbral VIEJOS (GT/10, MAX_THRESHOLD_DIST=10) — el piso trivial
# conocido en esa escala. NO se reescala aquí: el óptimo AUC-wise bajo el
# umbral nuevo (0..20) y target nuevo (GT/7.8125) no es necesariamente
# 1.28x este valor (el umbral no escaló por el mismo factor, 10->20 es 2x,
# no 1.28x), y recalcularlo requiere el dataset completo (solo en el pod).
# Se deja sin tocar a propósito: es un cambio fuera del alcance de este
# peldaño (afecta solo al ~10% de casos sin OCT). Ver INFORME.md, T1-M-86.
DISTANCE_FALLBACK_PX = 159.2

# Escala media del GT de Task 2 sobre los 1214 casos de entrenamiento
# (160 +- 13.7 px). Solo se usa como fallback si no hay volumen OCT.
SCALE_REF = 160.0

FUNDUS_TRAIN_SIZE = 1024  # tamaño con el que se entrenó; se reescala si difiere

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Bloques compartidos
# --------------------------------------------------------------------------

def _num_groups(channels: int) -> int:
    for g in range(min(8, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ¡OJO! Hay DOS implementaciones distintas de "dos convoluciones seguidas" en
# este proyecto, y NO son intercambiables: producen nombres de parámetro
# distintos, así que mezclarlas hace fallar `load_state_dict`.
#
#   _SeqConvBlock  -> self.block = nn.Sequential(...)   claves: block.0, block.1, ...
#                     lo usan Task1KeypointModel y FundusEnfaceHeatmapModel
#   _UNetConvBlock -> self.conv1 / self.gn1 / ...       claves: conv1, gn1, ...
#                     lo usa UNet (distancia)
#
# Unificarlas "porque son iguales" rompió esta submission una vez; lo cazó
# eval/run_local.py antes de subirla.

class _SeqConvBlock(nn.Module):
    """Variante con nn.Sequential — keypoint y registración."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        num_groups = _num_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UNetConvBlock(nn.Module):
    """Variante con atributos nombrados — solo el UNet de distancia."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        num_groups = _num_groups(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


def soft_argmax_2d(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Coordenadas (B, C, 2) en orden (x, y). Da sub-píxel gratis: el argmax duro
    sobre un heatmap de stride 16 pierde los primeros umbrales del AUC por
    cuantización, sin importar qué tan bueno sea el modelo."""
    batch, channels, height, width = heatmap.shape
    flat = heatmap.reshape(batch, channels, -1)
    weights = F.softmax(flat / temperature, dim=-1).reshape(batch, channels, height, width)
    ys = torch.linspace(0, height - 1, height, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.linspace(0, width - 1, width, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coord_x = (weights * grid_x).sum(dim=(-2, -1))
    coord_y = (weights * grid_y).sum(dim=(-2, -1))
    return torch.stack([coord_x, coord_y], dim=-1)


# --------------------------------------------------------------------------
# Task 1 — keypoint
# --------------------------------------------------------------------------

class _SimpleEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, n_downsamples: int = 4):
        super().__init__()
        self.in_conv = _SeqConvBlock(in_channels, base_channels)
        self.downsamples = nn.ModuleList()
        channels = base_channels
        for _ in range(n_downsamples):
            self.downsamples.append(
                nn.Sequential(nn.MaxPool2d(2), _SeqConvBlock(channels, channels * 2))
            )
            channels *= 2

    def forward(self, x):
        x = self.in_conv(x)
        for down in self.downsamples:
            x = down(x)
        return x


class Task1KeypointModel(nn.Module):
    def __init__(self, base_channels: int = 32, n_downsamples: int = 4,
                 heatmap_temperature: float = 1.0):
        super().__init__()
        self.encoder = _SimpleEncoder(3, base_channels, n_downsamples)
        feat_channels = base_channels * 2 ** n_downsamples
        self.heatmap_head = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.heatmap_temperature = heatmap_temperature

    def forward(self, fundus, fundus_size: int = FUNDUS_TRAIN_SIZE):
        feat = self.encoder(fundus)
        heatmap_logits = self.heatmap_head(feat)
        coords = soft_argmax_2d(heatmap_logits, temperature=self.heatmap_temperature)
        stride = fundus_size / heatmap_logits.shape[-1]
        keypoint = coords[:, 0, :] * stride
        return {"keypoint": keypoint, "heatmap_logits": heatmap_logits}


# --------------------------------------------------------------------------
# Task 1 — distancia (segmentación de B-scan + medición geométrica)
# --------------------------------------------------------------------------

class UNet(nn.Module):
    """0=fondo, 1=Ilm, 2=InstrumentInOCT. Se aplica por separado a cada B-scan."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(_UNetConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = out_ch
        self.bottleneck = _UNetConvBlock(ch, base_channels * (2 ** depth))
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_in = base_channels * (2 ** (i + 1))
            up_out = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2))
            self.decoders.append(_UNetConvBlock(up_out * 2, up_out))
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skips[self.depth - 1 - i]
            if x.shape[2:] != skip.shape[2:]:
                if x.shape[2] < skip.shape[2] or x.shape[3] < skip.shape[3]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                else:
                    skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.final_conv(x)


def distance_from_segmentation(seg_logits, ilm_class: int = 1, instrument_class: int = 2):
    """Punta = píxel de clase instrumento con MAYOR fila; ILM = fila mínima de
    esa clase en una ventana de +-5 columnas alrededor de la punta. Devuelve
    `ilm_row - tip_row`, o NaN si falta alguna de las dos clases."""
    batch_size = seg_logits.shape[0]
    seg = seg_logits.argmax(dim=1)
    distances = torch.full((batch_size,), float("nan"), dtype=torch.float32,
                            device=seg_logits.device)
    for b in range(batch_size):
        mask = seg[b]
        tip_mask = mask == instrument_class
        ilm_mask = mask == ilm_class
        if not tip_mask.any() or not ilm_mask.any():
            continue
        tip_rows, tip_cols = torch.nonzero(tip_mask, as_tuple=True)
        max_row_val, max_row_idx = tip_rows.max(dim=0)
        tip_row = max_row_val.item()
        tip_col = tip_cols[max_row_idx].item()
        col_min = max(0, tip_col - 5)
        col_max = min(mask.shape[1] - 1, tip_col + 5)
        ilm_in_window = ilm_mask[:, col_min:col_max + 1]
        if not ilm_in_window.any():
            continue
        ilm_rows = torch.nonzero(ilm_in_window, as_tuple=True)[0]
        distances[b] = float(ilm_rows.min().item() - tip_row)
    return distances


# --------------------------------------------------------------------------
# Task 2 — registración
# --------------------------------------------------------------------------

class FundusEnfaceHeatmapModel(nn.Module):
    def __init__(self, base_channels: int = 32, n_downsamples: int = 4,
                 heatmap_temperature: float = 1.0):
        super().__init__()
        self.fundus_encoder = _SimpleEncoder(3, base_channels, n_downsamples)
        self.enface_encoder = _SimpleEncoder(1, base_channels, n_downsamples)
        feat_channels = base_channels * 2 ** n_downsamples
        # 256 unidades, no 128: tiene que coincidir EXACTO con
        # src/fido/models/task2_baseline.py o los pesos no cargan.
        self.regression_head = nn.Sequential(
            nn.Linear(feat_channels * 2, 256), nn.ReLU(inplace=True), nn.Linear(256, 3)
        )
        self.heatmap_temperature = heatmap_temperature

    def forward(self, fundus, enface, fundus_size: int = FUNDUS_TRAIN_SIZE):
        # El volumen OCT cubre una región CUADRADA de retina pero se muestrea
        # anisótropo (128 slices x 512 A-scans). Sin este remuestreo la plantilla
        # sale 4:1 contra una huella que es 1:1 y el pico nunca es nítido.
        enface = F.interpolate(enface, size=(int(SCALE_REF), int(SCALE_REF)),
                                mode="bilinear", align_corners=False)
        fundus_feat = self.fundus_encoder(fundus)
        enface_feat = self.enface_encoder(enface)
        batch_size, channels, Hf, Wf = fundus_feat.shape
        _, _, Ht, Wt = enface_feat.shape

        pad_h, pad_w = Ht // 2, Wt // 2
        search = fundus_feat.reshape(1, batch_size * channels, Hf, Wf)
        corr = F.conv2d(search, enface_feat, groups=batch_size, padding=(pad_h, pad_w))
        out_h, out_w = Hf + 2 * pad_h - Ht + 1, Wf + 2 * pad_w - Wt + 1
        heatmap_logits = corr.reshape(batch_size, 1, out_h, out_w)

        heatmap_logits = heatmap_logits / (channels ** 0.5)
        mean = heatmap_logits.mean(dim=(2, 3), keepdim=True)
        std = heatmap_logits.std(dim=(2, 3), keepdim=True)
        heatmap_logits = (heatmap_logits - mean) / (std + 1e-6)

        coords = soft_argmax_2d(heatmap_logits, temperature=self.heatmap_temperature)

        fundus_vec = fundus_feat.mean(dim=(2, 3))
        enface_vec = enface_feat.mean(dim=(2, 3))
        cos_raw, sin_raw, scale_raw = self.regression_head(
            torch.cat([fundus_vec, enface_vec], dim=-1)
        ).unbind(dim=-1)
        norm = torch.sqrt(cos_raw ** 2 + sin_raw ** 2 + 1e-8)
        cos_theta = cos_raw / norm
        sin_theta = sin_raw / norm
        scale = SCALE_REF * torch.exp(scale_raw)

        # El pico marca el CENTRO de la plantilla; (tx,ty) es la esquina uv=(0,0).
        # Separación medida: 113.1 px sobre los 1214 casos reales. El -0.5 corrige
        # el sesgo de media celda de un kernel de correlación de lado par.
        stride = fundus_size / Hf
        center_x = (coords[:, 0, 0] - 0.5) * stride
        center_y = (coords[:, 0, 1] - 0.5) * stride
        tx = center_x - scale * (cos_theta + sin_theta) / 2.0
        ty = center_y - scale * (sin_theta - cos_theta) / 2.0
        return {"tx": tx, "ty": ty, "cos_theta": cos_theta,
                "sin_theta": sin_theta, "scale": scale}


def _compose_similarity(tx, ty, cos_theta, sin_theta, scale):
    """Similitud REFLEJADA: determinante negativo en el 100% de los 1214 casos
    de entrenamiento. No es una elección de diseño, es lo que dice el GT."""
    return np.array(
        [
            [scale * cos_theta, scale * sin_theta, tx],
            [scale * sin_theta, -scale * cos_theta, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# Preproceso — DEBE coincidir exactamente con el del entrenamiento
# --------------------------------------------------------------------------

def _fundus_to_tensor(opmi_image):
    """La ingestión entrega uint8 HWC sin normalizar; el dataset de
    entrenamiento hacía /255 y permute(2,0,1). Replicarlo exacto o el modelo ve
    entradas 255x fuera de rango."""
    arr = np.asarray(opmi_image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    tensor = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0)


def _enface_from_volume(oct_volume):
    """Proyección en-face: media sobre el eje de profundidad, normalizada a
    [0,1]. Réplica exacta de `enface_projection`, incluido el viaje por uint8
    (que cuantiza) para que el modelo vea lo mismo que en entrenamiento."""
    volume = np.asarray(oct_volume).astype(np.float32)
    projection = volume.mean(axis=1)
    projection = projection - projection.min()
    projection = projection / max(float(projection.max()), 1e-6)
    quantized = (projection * 255.0).astype(np.uint8)
    return torch.from_numpy(quantized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


# --------------------------------------------------------------------------
# API del contrato
# --------------------------------------------------------------------------

def load_model(model_path):
    """Recibe <submission>/model_0.pth o model_1.pth. El checkpoint de Task 1
    empaqueta DOS modelos (keypoint y distancia) porque el contrato solo admite
    un archivo de pesos por task."""
    payload = torch.load(str(model_path), map_location=_DEVICE)
    bundle = {"device": _DEVICE}

    if isinstance(payload, dict) and "keypoint" in payload:
        keypoint = Task1KeypointModel(base_channels=32, n_downsamples=4)
        keypoint.load_state_dict(payload["keypoint"])
        keypoint.to(_DEVICE).eval()
        bundle["keypoint"] = keypoint

        if payload.get("distance") is not None:
            unet = UNet(in_channels=1, num_classes=3, base_channels=32, depth=4)
            unet.load_state_dict(payload["distance"])
            unet.to(_DEVICE).eval()
            bundle["distance"] = unet
    else:
        registration = FundusEnfaceHeatmapModel(base_channels=32, n_downsamples=4)
        registration.load_state_dict(payload)
        registration.to(_DEVICE).eval()
        bundle["registration"] = registration

    return bundle


def _infer_task1(oct_volume, opmi_image, model):
    height, width = np.asarray(opmi_image).shape[:2]

    # --- keypoint ---
    keypoint_xy = [width / 2.0, height / 2.0]  # fallback: centro de la imagen
    net = model.get("keypoint")
    if net is not None:
        fundus = _fundus_to_tensor(opmi_image).to(model["device"])
        # El modelo se entrenó a 1024x1024. Si llega otro tamaño, se reescala y
        # las coordenadas se devuelven al espacio original.
        scale_x = width / float(FUNDUS_TRAIN_SIZE)
        scale_y = height / float(FUNDUS_TRAIN_SIZE)
        if fundus.shape[-2:] != (FUNDUS_TRAIN_SIZE, FUNDUS_TRAIN_SIZE):
            fundus = F.interpolate(fundus, size=(FUNDUS_TRAIN_SIZE, FUNDUS_TRAIN_SIZE),
                                    mode="bilinear", align_corners=False)
        with torch.no_grad():
            pred = net(fundus, fundus_size=FUNDUS_TRAIN_SIZE)
        xy = pred["keypoint"][0].detach().cpu().numpy()
        keypoint_xy = [float(xy[0] * scale_x), float(xy[1] * scale_y)]

    # --- distancia ---
    distance = DISTANCE_FALLBACK_PX
    unet = model.get("distance")
    if unet is not None and oct_volume is not None:
        volume = np.asarray(oct_volume)
        if volume.ndim == 3 and volume.shape[0] >= 1:
            slices = torch.from_numpy(volume.astype(np.float32) / 255.0)
            slices = slices.unsqueeze(1).to(model["device"])  # (n, 1, H, W)
            with torch.no_grad():
                logits = unet(slices)
                gaps = distance_from_segmentation(logits).detach().cpu().numpy()
            valid = gaps[~np.isnan(gaps)]
            if valid.size > 0:
                distance = float(DISTANCE_SCALE_A * float(valid.mean()) + DISTANCE_SCALE_B)

    return {"keypoints": [keypoint_xy[0], keypoint_xy[1]],
            "tool_tissue_distance": float(distance)}


def _infer_task2(oct_volume, opmi_image, model):
    height, width = np.asarray(opmi_image).shape[:2]
    net = model.get("registration")

    if net is None or oct_volume is None:
        # Sin volumen no hay plantilla que correlacionar. Fallback determinista:
        # escala media del entrenamiento, sin rotación, centrado en la imagen.
        return _compose_similarity(width / 2.0, height / 2.0, 1.0, 0.0, SCALE_REF)

    fundus = _fundus_to_tensor(opmi_image).to(model["device"])
    if fundus.shape[-2:] != (FUNDUS_TRAIN_SIZE, FUNDUS_TRAIN_SIZE):
        fundus = F.interpolate(fundus, size=(FUNDUS_TRAIN_SIZE, FUNDUS_TRAIN_SIZE),
                                mode="bilinear", align_corners=False)
    enface = _enface_from_volume(oct_volume).to(model["device"])

    with torch.no_grad():
        pred = net(fundus, enface, fundus_size=FUNDUS_TRAIN_SIZE)

    return _compose_similarity(
        float(pred["tx"][0]), float(pred["ty"][0]),
        float(pred["cos_theta"][0]), float(pred["sin_theta"][0]),
        float(pred["scale"][0]),
    )


def inference(task_id, oct_volume, opmi_image, model):
    """Punto de entrada del contrato.

    Envuelto por completo: una excepción aquí aborta los 100 casos, no solo
    este. Ante cualquier fallo se devuelve la predicción trivial, que puntúa
    bajo pero no destruye la corrida.
    """
    try:
        if int(task_id) == 0:
            return _infer_task1(oct_volume, opmi_image, model)
        return _infer_task2(oct_volume, opmi_image, model)
    except Exception:
        try:
            height, width = np.asarray(opmi_image).shape[:2]
        except Exception:
            height = width = float(FUNDUS_TRAIN_SIZE)
        if int(task_id) == 0:
            return {"keypoints": [width / 2.0, height / 2.0],
                    "tool_tissue_distance": DISTANCE_FALLBACK_PX}
        return _compose_similarity(width / 2.0, height / 2.0, 1.0, 0.0, SCALE_REF)
