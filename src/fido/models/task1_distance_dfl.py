"""T1-93: cabeza de distancia por bins (Distribution Focal Loss), replicando
la §2.2 de `experiments/87-organizer-baseline/INFORME.md` (Rohrmoser et al.,
arXiv 2603.25555 -- la cabeza de distancia de su YOLO-NAS+iOCT).

Por que existe: el pipeline geometrico actual (`unet_bscan_seg.UNet` +
`distance_from_segmentation`) SOLO produce una medicion cuando la
segmentacion encuentra a la vez la clase instrumento y la clase ILM en el
mismo B-scan. Cuando falla, cae al fallback constante (`DistanceCalibration.
fallback`). `analysis/measure_task1_segmentation_nan_fraction.py` (T1-93)
mide en que fraccion de casos reales eso ocurre -- ver
`experiments/93-t1-dfl/NAN_FRACTION_MEASUREMENT.md`. Una cabeza DFL es una
regresion directa desde imagen: SIEMPRE produce una distribucion, sin
depender de que la segmentacion intermedia encuentre nada.

Adaptacion deliberada frente al paper (documentada, no copiada a ciegas):
- El paper aplica DFL sobre mapas de features multi-escala de un detector
  denso (YOLO-NAS), con un logit por ancla espacial. Nuestro problema es
  regresion de UN escalar por caso (no hay deteccion densa ni anclas), asi
  que la cabeza opera sobre un vector de features ya global-average-pooled,
  no sobre un mapa espacial -- "dos bloques conv (1x1, 3x3) + BN/ReLU" del
  paper se traduce aqui a dos bloques Linear+BatchNorm1d+ReLU equivalentes.
- El encoder procesa los DOS B-scans ortogonales (00.png, 01.png) por
  separado con pesos COMPARTIDOS (mismo criterio que
  `unet_bscan_seg.UNet`, que tambien se aplica "de forma independiente a
  cada uno de los 2 B-scans de un frame"), y concatena ambos vectores de
  features antes de la cabeza -- preserva la identidad de cada B-scan
  (uno alineado con el eje del instrumento, el otro perpendicular) en vez
  de promediarlos a ciegas.
- `[d_min, d_max]` y `reg_max` NO son el `-1mm..6mm` del paper (unidades y
  sensor distintos) -- se eligen a partir de la distribucion real medida en
  `experiments/93-t1-dfl/DISTANCE_TARGET_HISTOGRAM.md` y se documentan con
  su justificacion en `experiments/93-t1-dfl/PRE_REGISTRATION.md`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(channels: int) -> int:
    """Mismo helper que `unet_bscan_seg.py`: mayor divisor de channels <= 8."""
    for g in range(min(8, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
    """Idem `unet_bscan_seg._ConvBlock` (GroupNorm, no BatchNorm -- estable
    con los batch sizes chicos que exige CPU/8GB de VRAM en local)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        num_groups = _get_num_groups(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class BscanSliceEncoder(nn.Module):
    """Encoder convolucional de UN B-scan (in_channels=1), pesos compartidos
    entre las 2 slices de un caso. Misma forma que `unet_bscan_seg.UNet`
    (sin decoder: solo el "brazo de bajada"), a proposito, para poder
    inicializar desde `encoders.*` de un checkpoint de segmentacion ya
    entrenado (`load_pretrained_segmentation_encoder`), aunque entrenar
    desde cero tambien funciona (ver tests)."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(_ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = out_ch
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            x = pool(x)
        return x  # (B, out_channels, H/2^depth, W/2^depth)


class DistanceDFLHead(nn.Module):
    """Softmax sobre `reg_max+1` bins + esperanza ponderada, reescalada
    linealmente a `[d_min, d_max]`. Traduccion a MLP de "dos bloques conv
    (1x1, 3x3) + BN/ReLU + conv 1x1 final" (INFORME.md Sec.2.2) para un
    vector de features ya pooled (ver docstring del modulo)."""

    def __init__(self, in_features: int, reg_max: int, d_min: float, d_max: float,
                 hidden: int = 128):
        super().__init__()
        if reg_max < 1:
            raise ValueError(f"reg_max must be >= 1, got {reg_max}")
        if d_max <= d_min:
            raise ValueError(f"d_max ({d_max}) must be > d_min ({d_min})")
        self.reg_max = reg_max
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(hidden, reg_max + 1)
        self.register_buffer("bin_values", torch.arange(reg_max + 1, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Devuelve los logits crudos `(B, reg_max+1)`, sin softmax (para
        entrenar con `dfl_loss`, que aplica log_softmax internamente)."""
        x = self.block1(features)
        x = self.block2(x)
        return self.out(x)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax -> esperanza ponderada -> reescalado lineal a
        `[d_min, d_max]`. `ŷ = d_min + E[bin]/reg_max * (d_max - d_min)`."""
        probs = F.softmax(logits, dim=-1)
        expected_bin = (probs * self.bin_values).sum(dim=-1)
        return self.d_min + expected_bin / self.reg_max * (self.d_max - self.d_min)

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        """Mapea un target fisico (px) a coordenada de bin continua en
        `[0, reg_max]`, RECORTADA a rango -- un target fuera de
        `[d_min, d_max]` no rompe nada, se satura al bin extremo mas
        cercano (gradiente sigue empujando hacia el borde, no explota)."""
        t = (target - self.d_min) / (self.d_max - self.d_min) * self.reg_max
        return t.clamp(min=0.0, max=float(self.reg_max))

    def target_distribution(self, target: torch.Tensor) -> torch.Tensor:
        """Distribucion "dos-calientes" (two-hot) que interpola entre los
        bins `floor(t)` y `ceil(t)` -- por construccion, su esperanza vale
        EXACTAMENTE `encode_target(target)` (identidad usada en el test de
        round-trip). Sirve para depurar/verificar `dfl_loss` sin acoplarse
        a la formula de cross-entropy."""
        t = self.encode_target(target)
        lo = t.floor().clamp(max=float(self.reg_max - 1) if self.reg_max > 0 else 0.0)
        lo_idx = lo.long()
        hi_idx = (lo_idx + 1).clamp(max=self.reg_max)
        weight_hi = (t - lo)
        weight_lo = 1.0 - weight_hi
        dist = torch.zeros(target.shape[0], self.reg_max + 1, device=target.device,
                            dtype=torch.float32)
        dist.scatter_(1, lo_idx.unsqueeze(1), weight_lo.unsqueeze(1))
        dist.scatter_add_(1, hi_idx.unsqueeze(1), weight_hi.unsqueeze(1))
        return dist


def dfl_loss(logits: torch.Tensor, target: torch.Tensor, head: DistanceDFLHead) -> torch.Tensor:
    """Distribution Focal Loss (Li et al. 2020, GFLv2): cross-entropy
    interpolada entre los dos bins que rodean el target continuo, en vez de
    cross-entropy contra un unico bin discreto -- empuja a la distribucion
    predicha a concentrarse (sharp) alrededor del valor real, no solo a que
    su esperanza acierte en promedio.

    `target` fuera de `[d_min, d_max]` se recorta dentro de `encode_target`
    (via `head`) antes de construir la perdida: nunca produce NaN/Inf, solo
    empuja el gradiente hacia el bin extremo correspondiente.
    """
    t = head.encode_target(target)
    lo = t.floor().clamp(max=float(head.reg_max - 1) if head.reg_max > 0 else 0.0)
    lo_idx = lo.long()
    hi_idx = (lo_idx + 1).clamp(max=head.reg_max)
    weight_hi = t - lo
    weight_lo = 1.0 - weight_hi

    log_probs = F.log_softmax(logits, dim=-1)
    log_p_lo = log_probs.gather(1, lo_idx.unsqueeze(1)).squeeze(1)
    log_p_hi = log_probs.gather(1, hi_idx.unsqueeze(1)).squeeze(1)
    loss = -(weight_lo * log_p_lo + weight_hi * log_p_hi)
    return loss.mean()


class Task1DistanceDFLModel(nn.Module):
    """Modelo completo: encoder de B-scan (pesos compartidos entre las 2
    slices) -> global average pool -> concat -> cabeza DFL."""

    def __init__(self, reg_max: int, d_min: float, d_max: float,
                 base_channels: int = 32, depth: int = 4, head_hidden: int = 128):
        super().__init__()
        self.encoder = BscanSliceEncoder(in_channels=1, base_channels=base_channels, depth=depth)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = DistanceDFLHead(
            in_features=self.encoder.out_channels * 2,  # 2 slices concatenadas
            reg_max=reg_max, d_min=d_min, d_max=d_max, hidden=head_hidden,
        )

    def forward(self, bscan: torch.Tensor) -> dict:
        """`bscan`: `(B, 2, H, W)` -- las 2 slices de un caso (mismo layout
        que `Task1Dataset`: `result["bscan"]` con forma `(2, size, size)`).
        Casos sin OCT llegan como ceros (ya los rellena `Task1Dataset`); el
        modelo igual produce una distribucion (no hay forma de que NaN'ee)."""
        if bscan.dim() != 4 or bscan.shape[1] != 2:
            raise ValueError(f"bscan must have shape (B, 2, H, W), got {tuple(bscan.shape)}")
        batch_size = bscan.shape[0]
        slices = bscan.reshape(batch_size * 2, 1, bscan.shape[2], bscan.shape[3])
        feat = self.encoder(slices)
        feat = self.pool(feat).reshape(batch_size, 2, -1)
        feat = feat.reshape(batch_size, -1)  # concat las 2 slices: (B, 2*out_channels)
        logits = self.head(feat)
        distance = self.head.decode(logits)
        return {"logits": logits, "distance": distance}


def load_pretrained_segmentation_encoder(encoder: BscanSliceEncoder, checkpoint_state: dict) -> int:
    """Inicializa `encoder.encoders.*` desde las claves `encoders.*` de un
    `state_dict` de `unet_bscan_seg.UNet` (p.ej. `raw["distance"]` de
    `submissions/r06-fallback-fixed/model_0.pth`) -- MISMA forma exacta
    (`_ConvBlock` con GroupNorm, mismo `base_channels`/`depth`), asi que las
    claves calzan 1:1 sin remapeo. Ignora `bottleneck.*`/`decoders.*`/
    `final_conv.*` (no existen en este encoder, que no tiene decoder).
    Devuelve cuantos tensores se copiaron, para poder verificarlo en el
    caller (0 copiados == las formas no coincidian, senal de bug silencioso)."""
    own_state = encoder.state_dict()
    matched = {k: v for k, v in checkpoint_state.items()
               if k.startswith("encoders.") and k in own_state and own_state[k].shape == v.shape}
    encoder.load_state_dict(matched, strict=False)
    return len(matched)
