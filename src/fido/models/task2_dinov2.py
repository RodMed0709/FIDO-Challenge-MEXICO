from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from fido.heatmap_decode import soft_argmax_2d
from fido.models.task2_baseline import SCALE_REF


class FundusEnfaceDinoV2Model(nn.Module):
    """Registración fundus/en-face con un DINOv2 siamés compartido."""

    def __init__(
        self,
        freeze_backbone: bool = True,
        proj_channels: int = 128,
        heatmap_temperature: float = 1.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        embed_dim = self.backbone.num_features
        self.projection = nn.Conv2d(embed_dim, proj_channels, kernel_size=1)
        self.regression_head = nn.Sequential(
            nn.Linear(proj_channels * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.heatmap_temperature = heatmap_temperature
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> FundusEnfaceDinoV2Model:
        super().train(mode)
        # Un backbone congelado debe conservar el comportamiento determinista
        # de evaluación aunque el bucle llame model.train() en cada época.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _feature_map(self, image: torch.Tensor, grid_size: int) -> torch.Tensor:
        image = (image - self.imagenet_mean) / self.imagenet_std
        tokens = self.backbone.forward_features(image)
        if isinstance(tokens, dict):
            tokens = tokens["x_norm_patchtokens"]
        else:
            tokens = tokens[:, self.backbone.num_prefix_tokens :]

        batch_size, num_tokens, channels = tokens.shape
        expected_tokens = grid_size * grid_size
        if num_tokens != expected_tokens:
            raise RuntimeError(
                f"DINOv2 produjo {num_tokens} tokens de parche; se esperaban {expected_tokens}"
            )
        feature_map = tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        return self.projection(feature_map)

    def forward(
        self, fundus: torch.Tensor, enface: torch.Tensor, fundus_size: int = 1024
    ) -> dict:
        """Predice la similitud que registra la proyección en-face en el fundus."""
        fundus = F.interpolate(fundus, size=(1022, 1022), mode="bilinear", align_corners=False)
        enface = F.interpolate(enface, size=(154, 154), mode="bilinear", align_corners=False)
        enface = enface.repeat(1, 3, 1, 1)

        fundus_feat = self._feature_map(fundus, grid_size=73)
        enface_feat = self._feature_map(enface, grid_size=11)

        batch_size, channels, fundus_h, fundus_w = fundus_feat.shape
        _, _, template_h, template_w = enface_feat.shape
        search = fundus_feat.reshape(1, batch_size * channels, fundus_h, fundus_w)
        corr = F.conv2d(
            search,
            enface_feat,
            groups=batch_size,
            padding=(template_h // 2, template_w // 2),
        )
        out_h = fundus_h + 2 * (template_h // 2) - template_h + 1
        out_w = fundus_w + 2 * (template_w // 2) - template_w + 1
        heatmap_logits = corr.reshape(batch_size, 1, out_h, out_w)
        heatmap_logits = heatmap_logits / math.sqrt(channels)
        mean = heatmap_logits.mean(dim=(2, 3), keepdim=True)
        std = heatmap_logits.std(dim=(2, 3), keepdim=True)
        heatmap_logits = (heatmap_logits - mean) / (std + 1e-6)

        coords = soft_argmax_2d(heatmap_logits, temperature=self.heatmap_temperature)
        fundus_vec = fundus_feat.mean(dim=(2, 3))
        enface_vec = enface_feat.mean(dim=(2, 3))
        cos_raw, sin_raw, scale_raw = self.regression_head(
            torch.cat([fundus_vec, enface_vec], dim=-1)
        ).unbind(dim=-1)

        norm = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-8)
        cos_theta = cos_raw / norm
        sin_theta = sin_raw / norm
        scale = SCALE_REF * torch.exp(scale_raw)

        stride = fundus_size / 73.0
        # La plantilla mide 11x11: con padding=11//2, la salida `o` representa
        # exactamente el centro `o`. Restar 0.5 introduciría un sesgo silencioso
        # de ~7 px (1024/73/2), necesario solo para kernels de lado par.
        center_x = coords[:, 0, 0] * stride
        center_y = coords[:, 0, 1] * stride

        # El pico localiza el centro de la plantilla, mientras que el GT guarda
        # la esquina uv=(0,0); la separación entre ambos puntos es A@(0.5,0.5).
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
        }


class _DinoDenseBranch(nn.Module):
    """Modality-specific DINOv2 branch with a lightweight stride-7 decoder."""

    def __init__(self, in_channels: int, descriptor_dim: int = 128,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1, bias=False),
            nn.GroupNorm(1, 3),
        )
        with torch.no_grad():
            self.input_adapter[0].weight.zero_()
            if in_channels == 3:
                for channel in range(3):
                    self.input_adapter[0].weight[channel, channel, 1, 1] = 1.0
            else:
                self.input_adapter[0].weight[:, 0, 1, 1] = 1.0
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=pretrained,
            num_classes=0, dynamic_img_size=True,
        )
        channels = self.backbone.num_features
        self.laterals = nn.ModuleList(nn.Conv2d(channels, descriptor_dim, 1) for _ in range(4))
        self.refine = nn.Sequential(
            nn.Conv2d(descriptor_dim, descriptor_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, descriptor_dim), nn.GELU(),
        )
        self.training_stage = "lp"

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Frozen blocks must not inject stochastic depth noise into LP.
            self.backbone.eval()
            if self.training_stage == "partial_ft":
                for block in self.backbone.blocks[-4:]:
                    block.train(True)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.input_adapter(image)
        features = self.backbone.forward_intermediates(
            image, indices=[2, 5, 8, 11], norm=True, output_fmt="NCHW",
            intermediates_only=True,
        )
        fused = sum(layer(feature) for layer, feature in zip(self.laterals, features)) / len(features)
        # Patch-14 tokens become an effective stride-7 dense grid.
        fused = F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False)
        return F.normalize(self.refine(fused), dim=1, eps=1e-6)


class DualDinoCommonEncoder(nn.Module):
    """Independent fundus/OCT DINOv2 branches; weights are never shared."""

    def __init__(self, descriptor_dim: int = 128, pretrained: bool = True,
                 fundus_input_size: int = 1022, oct_input_size: int = 224) -> None:
        super().__init__()
        self.fundus = _DinoDenseBranch(3, descriptor_dim, pretrained)
        self.oct = _DinoDenseBranch(1, descriptor_dim, pretrained)
        self.fundus_input_size = int(fundus_input_size)
        self.oct_input_size = int(oct_input_size)

    def forward(self, fundus: torch.Tensor, enface: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if valid_mask is None:
            valid_mask = torch.ones_like(fundus[:, :1], dtype=torch.bool)
        if valid_mask.shape != fundus[:, :1].shape:
            raise ValueError("valid_mask must have shape (B,1,H,W)")
        fundus_input = F.interpolate(fundus * valid_mask.to(fundus.dtype),
                                     (self.fundus_input_size,) * 2,
                                     mode="bilinear", align_corners=False)
        oct_input = F.interpolate(enface, (self.oct_input_size,) * 2,
                                  mode="bilinear", align_corners=False)
        fundus_desc = self.fundus(fundus_input)
        oct_desc = self.oct(oct_input)
        descriptor_valid = F.interpolate(valid_mask.float(), fundus_desc.shape[-2:],
                                         mode="nearest").bool()
        return {"fundus_desc": fundus_desc, "oct_desc": oct_desc,
                "descriptor_valid_mask": descriptor_valid}
