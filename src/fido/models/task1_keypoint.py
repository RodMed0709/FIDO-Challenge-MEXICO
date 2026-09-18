from __future__ import annotations

import torch
import torch.nn as nn

from fido.heatmap_decode import soft_argmax_2d


def _num_groups(channels: int) -> int:
    """Devuelve el mayor numero de grupos <= min(8, channels) que divide exactamente a channels."""
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    """Bloque convolucional: Conv2d 3x3 + GroupNorm + ReLU, repetido dos veces."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleEncoder(nn.Module):
    """Encoder CNN simple: convolucion inicial + bloques de downsample con maxpool y duplicacion de canales."""

    def __init__(self, in_channels: int, base_channels: int = 32, n_downsamples: int = 4):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, base_channels)
        self.downsamples = nn.ModuleList()
        channels = base_channels
        for _ in range(n_downsamples):
            self.downsamples.append(
                nn.Sequential(nn.MaxPool2d(2), ConvBlock(channels, channels * 2))
            )
            channels *= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for down in self.downsamples:
            x = down(x)
        return x


def heatmap_output_size(fundus_size: int, n_downsamples: int) -> int:
    """Tamano del heatmap de salida para una entrada cuadrada de fundus_size
    (asume division exacta por 2**n_downsamples). La usa el script de
    entrenamiento para saber el tamano del heatmap GT sin instanciar el modelo."""
    return fundus_size // (2 ** n_downsamples)


class Task1KeypointModel(nn.Module):
    """Localiza la punta de la canula en el fundus (RGB 1024x1024) via un encoder
    CNN + cabeza de heatmap de 1 canal decodificada con soft-argmax GLOBAL (no
    local -- un modelo sin entrenar necesita poder "saltar" a cualquier posicion
    del heatmap, una ventana local alrededor del argmax duro no lo permite)."""

    def __init__(self, base_channels: int = 32, n_downsamples: int = 4, heatmap_temperature: float = 1.0):
        super().__init__()
        # 3 canales de entrada: fundus RGB
        self.encoder = SimpleEncoder(3, base_channels, n_downsamples)
        feat_channels = base_channels * 2**n_downsamples
        # Cabeza de heatmap de 1 canal, convolucion 1x1 pura (sin activacion --
        # el BCE de entrenamiento se aplica sobre logits crudos via
        # binary_cross_entropy_with_logits, asi que NO debe llevar sigmoid aqui)
        self.heatmap_head = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.heatmap_temperature = heatmap_temperature

    def forward(self, fundus: torch.Tensor, fundus_size: int = 1024) -> dict:
        # fundus: (B, 3, H, W) con H=W=fundus_size (imagen cuadrada)
        feat = self.encoder(fundus)  # (B, feat_channels, Hh, Ww)
        heatmap_logits = self.heatmap_head(feat)  # (B, 1, Hh, Ww)
        coords = soft_argmax_2d(heatmap_logits, temperature=self.heatmap_temperature)  # (B, 1, 2), (x, y) en grilla del heatmap
        # La entrada es cuadrada, asi que alto y ancho del heatmap coinciden
        stride = fundus_size / heatmap_logits.shape[-1]
        keypoint = coords[:, 0, :] * stride  # (B, 2), (x, y) en pixeles reales del fundus
        return {"keypoint": keypoint, "heatmap_logits": heatmap_logits}
