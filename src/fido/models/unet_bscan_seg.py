from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(channels: int) -> int:
    """Devuelve el mayor divisor de channels que sea <= 8."""
    for g in range(min(8, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
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


class UNet(nn.Module):
    """Segmentador liviano de B-scan (T1-R5): 0=fondo, 1=Ilm, 2=InstrumentInOCT.
    Se aplica de forma independiente a cada uno de los 2 B-scans de un frame."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        depth: int = 4,
    ):
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

        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = _ConvBlock(ch, bottleneck_ch)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_in = base_channels * (2 ** (i + 1))
            up_out = base_channels * (2**i)
            self.upconvs.append(nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2))
            self.decoders.append(_ConvBlock(up_out * 2, up_out))

        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[self.depth - 1 - i]

            if x.shape[2:] != skip.shape[2:]:
                if x.shape[2] < skip.shape[2] or x.shape[3] < skip.shape[3]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                else:
                    skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final_conv(x)


def distance_from_segmentation(
    seg_logits: torch.Tensor,
    ilm_class: int = 1,
    instrument_class: int = 2,
) -> torch.Tensor:
    """Réplica batched de `tip_and_ilm_column` + `ilm_row_near_column`
    (analysis/verify_task1_distance_geometry.py, NumPy caso por caso) sobre
    logits de segmentación. Punta = píxel de la clase instrumento con mayor
    fila; ILM = fila mínima de esa clase en una ventana de ±5 columnas
    alrededor de la columna de la punta. Devuelve `ilm_row - tip_row` (B,),
    NaN para casos sin píxeles de alguna clase (no rompe el batch)."""
    batch_size = seg_logits.shape[0]
    seg = seg_logits.argmax(dim=1)  # (B, H, W)

    distances = torch.full((batch_size,), float("nan"), dtype=torch.float32, device=seg_logits.device)

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
        min_ilm_row = ilm_rows.min().item()

        distances[b] = float(min_ilm_row - tip_row)

    return distances
