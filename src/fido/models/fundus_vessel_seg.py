from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(channels: int) -> int:
    for g in range(min(8, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = _get_num_groups(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class FundusVesselUNet(nn.Module):
    """Segmentador binario de vasos en el fundus RGB (T2-R4, motor de
    segmentación previo al puente por matching clásico).

    Salida: UN SOLO canal de logits crudos (B, 1, H, W), sin sigmoid aplicado
    dentro del forward. Es un problema binario (vaso vs fondo), no
    multi-clase como el segmentador de B-scan (`unet_bscan_seg.py`, que sí
    necesita softmax de 3 clases porque fondo/Ilm/instrumento son mutuamente
    excluyentes con más de 2 categorías) — un solo canal + sigmoid afuera (en
    la loss combinada `combined_bce_dice_loss` o en inferencia) es más simple
    y evita normalizar una distribución de 2 clases que es simétrica por
    construcción (P(vaso) = 1 - P(fondo)).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = in_channels
        for i in range(depth):
            self.encoders.append(_ConvBlock(ch, base_channels * (2 ** i)))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            ch = base_channels * (2 ** i)

        self.bottleneck = _ConvBlock(ch, ch * 2)

        # in_channels de cada ConvTranspose2d: SIEMPRE base_channels*2**(i+1),
        # tanto para la primera etapa (que recibe el bottleneck, con canales
        # base*2**depth) como para las siguientes (que reciben la salida del
        # decoder anterior, con canales base*2**(i+1) por construcción — es
        # la misma fórmula porque out_ch de la etapa i+1 es justo
        # base*2**(i+1)). Un `ch` mutable multiplicado por 2 en cada
        # iteración da el valor correcto SOLO en la primera etapa (por
        # coincidencia con el bottleneck) y se duplica de más en las
        # siguientes -> mismatch de canales en el ConvTranspose2d. Se usa la
        # fórmula cerrada, no un contador con estado, para evitar ese bug.
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            in_ch = base_channels * (2 ** (i + 1))
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(_ConvBlock(out_ch * 2, out_ch))

        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            skips.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)

        x = self.bottleneck(x)

        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            skip = skips[self.depth - 1 - i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final_conv(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Calcula 1 - Dice coefficient medio (por batch) para segmentación binaria.

    Aplica sigmoid a los logits para obtener probabilidades, luego calcula
    el Dice coefficient entre la máscara predicha y la real. La pérdida es
    1 - Dice, promediada sobre el batch.

    Args:
        logits: Logits crudos de salida (B, 1, H, W).
        target: Máscara binaria real (B, 1, H, W), valores {0, 1}.
        eps: Épsilon para estabilidad numérica.

    Returns:
        Pérdida Dice escalar.
    """
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    target = target.reshape(target.size(0), -1)

    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def combined_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """Combina BCE-with-logits (pos_weight) y Dice loss para vasos finos.

    BCE con pos_weight proporciona señal por-píxel con frecuencia inversa,
    contrarrestando el desbalance severo (pocos píxeles positivos).
    Dice loss optimiza directamente el solapamiento de la estructura delgada
    minoritaria, que es exactamente lo que interesa en la segmentación de
    vasos (igual que en el problema de cánula con ~0.008% de píxeles positivos
    en Task 1 de este proyecto, donde CrossEntropyLoss sin pesos colapsó a
    predecir fondo siempre — ver ATTACK_LADDER.md, T1-R5).

    Args:
        logits: Logits crudos (B, 1, H, W).
        target: Máscara binaria real (B, 1, H, W), valores {0, 1}.
        pos_weight: Peso para la clase positiva (vaso vs fondo). Acepta un
            float/int plano (se convierte a tensor automáticamente) o un
            tensor ya construido — `F.binary_cross_entropy_with_logits` NO
            acepta un float crudo (lanza TypeError en tiempo de ejecución:
            "pos_weight must be a Tensor"), así que la conversión es
            necesaria, no cosmética.
        dice_weight: Peso relativo de la componente Dice.

    Returns:
        Pérdida escalar combinada.
    """
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = dice_loss(logits, target)
    return bce + dice_weight * dice


def vessel_pixel_fraction(mask_batch: torch.Tensor) -> float:
    """Calcula la fracción de píxeles de vaso (positivos) en un batch.

    Args:
        mask_batch: Batch de máscaras binarias, shape (B, 1, H, W) o (B, H, W).

    Returns:
        Fracción de píxeles positivos en todo el batch.
    """
    if mask_batch.dim() == 4 and mask_batch.size(1) == 1:
        mask_flat = mask_batch.view(-1)
    else:
        mask_flat = mask_batch.reshape(-1)
    return (mask_flat > 0.5).float().mean().item()
