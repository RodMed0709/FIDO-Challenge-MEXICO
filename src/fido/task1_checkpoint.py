from __future__ import annotations

from pathlib import Path

import torch

from fido.models.task1_keypoint_resnet import Task1KeypointResNet18FPN

SCHEMA = "fido.task1.keypoint.v1"


def make_task1_keypoint_payload(model, *, architecture: str, fold: int, seed: int,
                                epoch: int, model_kwargs: dict | None = None) -> dict:
    return {
        "schema": SCHEMA,
        "architecture": architecture,
        "model_kwargs": dict(model_kwargs or {}),
        "state_dict": model.state_dict(),
        "provenance": {"rung": "T1-85", "fold": fold, "seed": seed, "epoch": epoch},
    }


def load_task1_keypoint_payload(path: str | Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported Task 1 keypoint checkpoint schema")
    if payload.get("architecture") != "resnet18-fpn":
        raise ValueError(f"Unsupported architecture: {payload.get('architecture')}")
    kwargs = payload.get("model_kwargs", {})
    model = Task1KeypointResNet18FPN(pretrained=False, **kwargs)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
