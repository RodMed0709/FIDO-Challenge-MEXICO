from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from fido.heatmap_decode import soft_argmax_2d


class Task1KeypointDinoV2Model(nn.Module):
    """Localiza la punta de la canula con DINOv2 y una cabeza de heatmap.

    La grilla nativa de 73x73 equivale a celdas de unos 14 px. El upsample 2x
    produce 146x146 (unos 7 px por celda sobre 1024 px), frente a los 16 px de
    la CNN que alcanzo AUC=0.8536 y error medio=1.03 px en 14,399 casos.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        upsample_factor: int = 2,
        heatmap_temperature: float = 1.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if upsample_factor < 1:
            raise ValueError("upsample_factor debe ser un entero mayor o igual que 1")

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
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=1),
            nn.GELU(),
            nn.Upsample(
                scale_factor=upsample_factor,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # Logits crudos: la perdida aplica binary_cross_entropy_with_logits.
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.heatmap_temperature = heatmap_temperature
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> Task1KeypointDinoV2Model:
        super().train(mode)
        # Evita que model.train() altere un encoder que debe permanecer congelado.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _feature_map(self, fundus: torch.Tensor) -> torch.Tensor:
        fundus = (fundus - self.imagenet_mean) / self.imagenet_std
        tokens = self.backbone.forward_features(fundus)
        if isinstance(tokens, dict):
            tokens = tokens["x_norm_patchtokens"]
        else:
            tokens = tokens[:, self.backbone.num_prefix_tokens :]

        batch_size, num_tokens, channels = tokens.shape
        if num_tokens != 73 * 73:
            raise RuntimeError(
                f"DINOv2 produjo {num_tokens} tokens de parche; se esperaban {73 * 73}"
            )
        return tokens.transpose(1, 2).reshape(batch_size, channels, 73, 73)

    def forward(self, fundus: torch.Tensor, fundus_size: int = 1024) -> dict:
        """Devuelve el keypoint en pixeles y los logits del heatmap."""
        # 1022=73*14 evita truncar parches y fija una grilla cuadrada de 73x73.
        fundus = F.interpolate(
            fundus, size=(1022, 1022), mode="bilinear", align_corners=False
        )
        feature_map = self._feature_map(fundus)
        heatmap_logits = self.heatmap_head(feature_map)
        coords = soft_argmax_2d(
            heatmap_logits, temperature=self.heatmap_temperature
        )
        # El stride debe reflejar la resolucion real tras el upsample (146 por defecto).
        stride = fundus_size / heatmap_logits.shape[-1]
        keypoint = coords[:, 0, :] * stride
        return {"keypoint": keypoint, "heatmap_logits": heatmap_logits}
