from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fido.heatmap_decode import soft_argmax_2d

# Escala de referencia para parametrizar la cabeza de regresion: media medida
# sobre los 1214 casos reales de entrenamiento (160 +- 13.7 px, ver T2-R1 en
# ATTACK_LADDER.md). Es solo el punto de partida de una prediccion
# multiplicativa -- el modelo puede llegar a cualquier escala positiva, incluida
# la del Mock Test (~215 px), que cae fuera del rango de entrenamiento.
SCALE_REF = 160.0


def _num_groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    """Dos capas de Conv2d + GroupNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleEncoder(nn.Module):
    """Encoder con max-pooling y duplicación de canales. No asume entrada
    cuadrada — funciona igual sobre el fundus (1024x1024) y el en-face
    (128x512, aspect ratio distinto)."""

    def __init__(self, in_channels: int, base_channels: int = 32, n_downsamples: int = 4) -> None:
        super().__init__()
        self.in_conv = ConvBlock(in_channels, base_channels)
        self.downsamples = nn.ModuleList()
        channels = base_channels
        for _ in range(n_downsamples):
            self.downsamples.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ConvBlock(channels, channels * 2),
                )
            )
            channels *= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for down in self.downsamples:
            x = down(x)
        return x


class FundusEnfaceHeatmapModel(nn.Module):
    """Baseline T2-R2: correlación cruzada espacial real (estilo SiamFC) entre
    el mapa de features del fundus y el del en-face — reemplaza un primer
    intento con FiLM que se estancó en ~65px incluso memorizando 5 ejemplos
    (FiLM solo pasa un vector GLOBAL del en-face, sin ubicación espacial)."""

    def __init__(self, base_channels: int = 32, n_downsamples: int = 4,
                 heatmap_temperature: float = 1.0) -> None:
        super().__init__()
        self.fundus_encoder = SimpleEncoder(3, base_channels, n_downsamples)
        self.enface_encoder = SimpleEncoder(1, base_channels, n_downsamples)
        feat_channels = base_channels * 2**n_downsamples
        self.regression_head = nn.Sequential(
            nn.Linear(feat_channels * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.heatmap_temperature = heatmap_temperature

    def forward(self, fundus: torch.Tensor, enface: torch.Tensor, fundus_size: int = 1024,
                valid_mask: torch.Tensor | None = None) -> dict:
        """
        fundus: (B, 3, Hf, Wf) con Hf=Wf=1024
        enface: (B, 1, Ht, Wt) con Ht=128, Wt=512
        """
        # El volumen OCT cubre una region CUADRADA de la retina (el GT es una
        # similitud: ortogonalidad verificada al 100% sobre los 1214 casos), pero
        # se muestrea de forma anisotropa: 128 slices x 512 A-scans. Tal cual, la
        # plantilla sale de 8x32 celdas (relacion 4:1) mientras que la huella real
        # en el fundus es un cuadrado de lado ~160 px = 10x10 celdas. La
        # correlacion cruzada estaba comparando dos geometrias incompatibles
        # (2.4x demasiado ancha, 1.25x demasiado corta) y por eso el pico nunca
        # podia ser nitido. Se remuestrea a SCALE_REF x SCALE_REF para que, tras
        # los 4 downsamples del encoder, la plantilla salga en la misma escala
        # que la huella que debe encontrar.
        enface = F.interpolate(
            enface, size=(int(SCALE_REF), int(SCALE_REF)), mode="bilinear", align_corners=False
        )

        has_explicit_mask = valid_mask is not None
        if valid_mask is None:
            valid_mask = torch.ones_like(fundus[:, :1], dtype=torch.bool)
        if valid_mask.shape != fundus[:, :1].shape:
            raise ValueError("valid_mask must have shape (B,1,Hf,Wf)")
        # Invalid/padded pixels are removed before the encoder, so their value
        # cannot become a photometric shortcut.
        fundus = fundus * valid_mask.to(fundus.dtype)
        fundus_feat = self.fundus_encoder(fundus)  # (B, C, 64, 64)
        enface_feat = self.enface_encoder(enface)  # (B, C, 10, 10)

        batch_size, channels, Hf, Wf = fundus_feat.shape
        _, _, Ht, Wt = enface_feat.shape

        # Correlación cruzada por lote (truco de conv agrupada, estilo SiamFC):
        # groups=batch_size hace que cada grupo del input (cada muestra del
        # batch) solo vea su propio kernel (la plantilla en-face de la misma
        # muestra), evitando que se crucen muestras entre sí.
        #
        # padding=(Ht//2, Wt//2), NO conv "válida" (sin padding): una revisión
        # adversarial midió contra los 1214 casos reales de Task 2 que la
        # versión sin padding es estructuralmente incapaz de representar
        # cualquier tx fuera de [256,768]px — el template (mitad de ancho del
        # mapa de búsqueda) nunca puede centrarse cerca del borde. Eso pisaba
        # el AUC en ~3.87% de los casos reales, invisible en el smoke test (1
        # escenario, ningún caso cayó en esa cola). Con este padding el centro
        # del template SÍ puede caer en cualquier celda de fundus_feat,
        # incluidos los bordes.
        pad_h, pad_w = Ht // 2, Wt // 2
        search = fundus_feat.reshape(1, batch_size * channels, Hf, Wf)
        kernel = enface_feat  # (batch_size, channels, Ht, Wt) sirve tal cual como weight
        corr = F.conv2d(search, kernel, groups=batch_size, padding=(pad_h, pad_w))
        out_h, out_w = Hf + 2 * pad_h - Ht + 1, Wf + 2 * pad_w - Wt + 1
        heatmap_logits = corr.reshape(batch_size, 1, out_h, out_w)

        feature_valid = F.interpolate(valid_mask.float(), size=(Hf, Wf), mode="nearest")
        if has_explicit_mask:
            valid_search = feature_valid.reshape(1, batch_size, Hf, Wf)
            valid_kernel = torch.ones((batch_size, 1, Ht, Wt), device=fundus.device,
                                      dtype=fundus.dtype)
            support_count = F.conv2d(valid_search, valid_kernel, groups=batch_size,
                                     padding=(pad_h, pad_w))
            heatmap_valid = support_count.reshape(batch_size, 1, out_h, out_w) >= (Ht * Wt - 1e-4)
        else:
            # Preserve the historical baseline exactly: its deliberate SiamFC
            # padding permits centers at the fundus boundary. Only augmented
            # samples carry an explicit geometric invalid mask.
            heatmap_valid = torch.ones_like(heatmap_logits, dtype=torch.bool)

        # La correlación cruda puede salir en cualquier escala dependiendo de
        # la magnitud de los features en cada paso de entrenamiento — dividir
        # por sqrt(channels) NO basta (un smoke test mostró logits ya en las
        # centenas/miles desde la época 1). Acotar con clamp o tanh tampoco
        # sirve: ambos tienen gradiente ~cero una vez el valor de entrada ya
        # es enorme (tanh(x) redondea a 1.0 exacto en float32 para x grande,
        # mismo problema que un clamp duro). La solución real es normalizar
        # la ESCALA, no recortar el valor: restar la media y dividir por la
        # desviación estándar de cada mapa dejan el heatmap siempre en una
        # escala razonable (tipo instance norm), sin importar qué tan grande
        # salga la correlación cruda, y sin dejar de ser diferenciable.
        heatmap_logits = heatmap_logits / (channels ** 0.5)
        weights = heatmap_valid.to(heatmap_logits.dtype)
        count = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (heatmap_logits * weights).sum(dim=(2, 3), keepdim=True) / count
        if has_explicit_mask:
            variance = ((heatmap_logits - mean).square() * weights).sum(
                dim=(2, 3), keepdim=True
            ) / count
            std = torch.sqrt(variance)
        else:
            # Match the pre-T2-81 baseline's corrected sample std exactly.
            std = heatmap_logits.std(dim=(2, 3), keepdim=True)
        heatmap_logits = (heatmap_logits - mean) / (std + 1e-6)
        heatmap_logits = heatmap_logits.masked_fill(~heatmap_valid, -1e4)

        coords = soft_argmax_2d(heatmap_logits, temperature=self.heatmap_temperature)  # (B,1,2) en (x,y)

        fundus_weights = feature_valid.to(fundus_feat.dtype)
        fundus_vec = (fundus_feat * fundus_weights).sum(dim=(2, 3)) / fundus_weights.sum(
            dim=(2, 3)
        ).clamp_min(1.0)
        enface_vec = enface_feat.mean(dim=(2, 3))
        reg_input = torch.cat([fundus_vec, enface_vec], dim=-1)
        cos_raw, sin_raw, scale_raw = self.regression_head(reg_input).unbind(dim=-1)

        norm = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-8)
        cos_theta = cos_raw / norm
        sin_theta = sin_raw / norm
        # Escala parametrizada como SCALE_REF*exp(raw), no softplus(raw)+1:
        # softplus arrancaba en 1.69 contra un objetivo real de ~160, asi que el
        # termino de escala del MSE valia ~45 (el 97% de la perdida total) y se
        # comia todo el presupuesto de clip_grad_norm_. Con esta forma raw=0 da
        # exactamente SCALE_REF y el error queda multiplicativo, no absoluto.
        scale = SCALE_REF * torch.exp(scale_raw)

        # El pico de la correlacion cruzada marca donde se alinea el CENTRO de
        # la plantilla en-face sobre el fundus. Pero (tx,ty) del GT es la esquina
        # uv=(0,0) del cuadrado unitario, no el centro. Son dos puntos separados
        # por A@(0.5,0.5), es decir s*sqrt(2)/2 = 113.1 px de media medidos sobre
        # los 1214 casos reales -- exactamente del orden del error de validacion
        # que tenia este modelo (98.74 px). Asignar el pico directo a (tx,ty)
        # ponia un techo de AUC en 0.031 aunque la localizacion fuera perfecta.
        #
        # El -0.5 corrige un sesgo aparte: el kernel de correlacion tiene lados
        # PARES (Ht=8, Wt=32), asi que con padding=(Ht//2, Wt//2) la celda de
        # salida `o` corresponde al centro de plantilla `o - 0.5`, no a `o`.
        # Son 0.5 celdas x 16 px = 8 px por eje, 11.3 px de error de esquina:
        # por si solo pasaba del umbral de 10 px del AUC.
        stride = fundus_size / Hf
        center_x = (coords[:, 0, 0] - 0.5) * stride
        center_y = (coords[:, 0, 1] - 0.5) * stride
        tx = center_x - scale * (cos_theta - sin_theta) / 2.0
        ty = center_y - scale * (sin_theta + cos_theta) / 2.0

        return {
            "tx": tx,
            "ty": ty,
            "center_x": center_x,
            "center_y": center_y,
            "cos_theta": cos_theta,
            "sin_theta": sin_theta,
            "scale": scale,
            "heatmap_logits": heatmap_logits,
            "heatmap_valid_mask": heatmap_valid,
        }


def heatmap_output_size(
    fundus_size: int,
    base_channels: int,
    n_downsamples: int,
    enface_height: int = 128,
    enface_width: int = 512,
) -> tuple[int, int]:
    """Calcula (alto, ancho) del heatmap de correlación sin instanciar el
    modelo — lo usa el script de entrenamiento para generar el heatmap GT.
    Con padding=(Ht//2, Wt//2) (ver forward()), no conv "válida"."""
    del base_channels
    Hf = Wf = fundus_size // (2**n_downsamples)
    Ht = enface_height // (2**n_downsamples)
    Wt = enface_width // (2**n_downsamples)
    pad_h, pad_w = Ht // 2, Wt // 2
    return (Hf + 2 * pad_h - Ht + 1, Wf + 2 * pad_w - Wt + 1)
