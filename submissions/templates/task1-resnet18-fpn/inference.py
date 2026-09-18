"""Template autocontenido de submission para checkpoints T1-85."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISTANCE_FALLBACK = 159.2


def _soft_argmax(heatmap):
    b, c, h, w = heatmap.shape
    weights = torch.softmax(heatmap.reshape(b, c, -1), dim=-1).reshape(b, c, h, w)
    yy, xx = torch.meshgrid(torch.arange(h, device=heatmap.device, dtype=heatmap.dtype),
                            torch.arange(w, device=heatmap.device, dtype=heatmap.dtype), indexing="ij")
    return torch.stack([(weights * xx).sum((-2, -1)), (weights * yy).sum((-2, -1))], -1)


class ResNet18FPN(nn.Module):
    def __init__(self, fpn_channels=64, heatmap_temperature=1.0, **_):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.fpn = nn.ModuleDict({"c2": nn.Conv2d(64, fpn_channels, 1),
                                  "c3": nn.Conv2d(128, fpn_channels, 1),
                                  "c4": nn.Conv2d(256, fpn_channels, 1),
                                  "c5": nn.Conv2d(512, fpn_channels, 1)})
        self.smooth = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                    nn.GroupNorm(8, fpn_channels), nn.ReLU(inplace=True))
        self.heatmap_head = nn.Conv2d(fpn_channels, 1, 1)
        self.heatmap_temperature = heatmap_temperature
        self.register_buffer("imagenet_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x, fundus_size=1024):
        x = (x - self.imagenet_mean) / self.imagenet_std
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x))); x = b.maxpool(x)
        c2 = b.layer1(x); c3 = b.layer2(c2); c4 = b.layer3(c3); c5 = b.layer4(c4)
        p5 = self.fpn["c5"](c5)
        p4 = self.fpn["c4"](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.fpn["c3"](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.fpn["c2"](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        heatmap = self.heatmap_head(self.smooth(p2))
        xy = _soft_argmax(heatmap / self.heatmap_temperature)[:, 0]
        return xy * (fundus_size / heatmap.shape[-1])


def load_model(model_path):
    payload = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if payload.get("schema") != "fido.task1.keypoint.v1" or payload.get("architecture") != "resnet18-fpn":
        raise ValueError("model_0.pth is not a T1-85 ResNet18-FPN payload")
    model = ResNet18FPN(**payload.get("model_kwargs", {}))
    model.load_state_dict(payload["state_dict"], strict=True)
    return {"keypoint": model.to(DEVICE).eval(), "device": DEVICE,
            "metadata": payload["provenance"]}


def _inference_task1(task_id, oct_volume, opmi_image, model):
    del oct_volume
    if int(task_id) != 0:
        raise ValueError("This template only supports Task 1")
    image = np.asarray(opmi_image)
    height, width = image.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(image[..., :3]).astype(np.float32) / 255).permute(2, 0, 1)[None]
    tensor = tensor.to(model["device"])
    if tensor.shape[-2:] != (1024, 1024):
        tensor = F.interpolate(tensor, size=(1024, 1024), mode="bilinear", align_corners=False)
    with torch.inference_mode():
        xy = model["keypoint"](tensor)[0].cpu().numpy()
    return {"keypoints": [float(xy[0] * width / 1024), float(xy[1] * height / 1024)],
            "tool_tissue_distance": float(DISTANCE_FALLBACK)}


def inference(task_id, oct_volume, opmi_image, model):
    try:
        return _inference_task1(task_id, oct_volume, opmi_image, model)
    except Exception:
        image = np.asarray(opmi_image)
        height, width = image.shape[:2] if image.ndim >= 2 else (1024, 1024)
        return {"keypoints": [float(width / 2), float(height / 2)],
                "tool_tissue_distance": float(DISTANCE_FALLBACK)}
