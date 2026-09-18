import importlib.util
from pathlib import Path

import numpy as np
import torch

from fido.models.task1_keypoint_resnet import Task1KeypointResNet18FPN
from fido.train.train_task1_keypoint import (
    accumulation_window_size,
    build_optimizer_param_groups,
    ledger_identity,
    scale_loss_for_accumulation,
    should_optimizer_step,
)
from fido.task1_checkpoint import load_task1_keypoint_payload, make_task1_keypoint_payload


def test_resnet_fpn_outputs_stride_four_heatmap_and_image_coordinates():
    model = Task1KeypointResNet18FPN(pretrained=False)
    torch.nn.init.zeros_(model.heatmap_head.weight)
    torch.nn.init.zeros_(model.heatmap_head.bias)
    output = model(torch.zeros(1, 3, 128, 128), fundus_size=128)
    assert output["heatmap_logits"].shape == (1, 1, 32, 32)
    assert output["keypoint"].shape == (1, 2)
    assert torch.allclose(output["keypoint"], torch.tensor([[62.0, 62.0]]), atol=1e-4)


def test_resnet_fpn_production_resolution_is_256_square():
    model = Task1KeypointResNet18FPN(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 1024, 1024))
    assert output["heatmap_logits"].shape == (1, 1, 256, 256)


def test_resnet_fpn_state_dict_loads_without_pretrained_download():
    source = Task1KeypointResNet18FPN(pretrained=False)
    target = Task1KeypointResNet18FPN(pretrained=False)
    target.load_state_dict(source.state_dict(), strict=True)
    assert torch.equal(target.backbone.conv1.weight, source.backbone.conv1.weight)


def test_frozen_resnet_stays_eval_and_only_head_is_trainable():
    model = Task1KeypointResNet18FPN(pretrained=False, freeze_backbone=True)
    model.train()
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.fpn.parameters())


def test_discriminative_lr_separates_backbone_and_decoder():
    model = Task1KeypointResNet18FPN(pretrained=False, freeze_backbone=False)
    groups = build_optimizer_param_groups(model, lr=1e-4, backbone_lr_multiplier=0.1)
    assert sorted(group["lr"] for group in groups) == [1e-5, 1e-4]
    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert grouped == trainable


def test_accumulation_steps_full_windows_and_final_remainder():
    assert [i for i in range(10) if should_optimizer_step(i, 10, 4)] == [3, 7, 9]
    assert [accumulation_window_size(i, 10, 4) for i in range(10)] == [4] * 8 + [2, 2]


def test_accumulation_default_preserves_historical_step_count():
    assert all(should_optimizer_step(i, 3, 1) for i in range(3))
    assert all(accumulation_window_size(i, 3, 1) == 1 for i in range(3))


def test_accumulation_scales_full_and_remainder_windows_correctly():
    loss = torch.tensor(8.0)
    assert scale_loss_for_accumulation(loss, 0, 10, 4).item() == 2.0
    assert scale_loss_for_accumulation(loss, 8, 10, 4).item() == 4.0
    assert scale_loss_for_accumulation(loss, 0, 10, 1).item() == 8.0


def test_t1_85_ledger_identity_distinguishes_candidate_and_matched_control():
    assert ledger_identity("resnet18-fpn", True, 4)[:2] == (
        "t1-85-resnet-fpn", "85-t1-cnn-fpn")
    assert ledger_identity("cnn", True, 4)[:2] == (
        "t1-85-cnn-control", "85-t1-cnn-fpn-control")


def test_resnet_checkpoint_roundtrip_and_submission_smoke(tmp_path):
    model = Task1KeypointResNet18FPN(pretrained=False).eval()
    payload = make_task1_keypoint_payload(
        model, architecture="resnet18-fpn", fold=0, seed=0, epoch=3,
        model_kwargs={"fpn_channels": 64, "heatmap_temperature": 1.0},
    )
    checkpoint = tmp_path / "model_0.pth"
    torch.save(payload, checkpoint)
    restored, metadata = load_task1_keypoint_payload(checkpoint)
    sample = torch.zeros(1, 3, 128, 128)
    with torch.inference_mode():
        expected = model(sample, fundus_size=128)["keypoint"]
        actual = restored(sample, fundus_size=128)["keypoint"]
    assert torch.equal(actual, expected)
    assert metadata["provenance"] == {"rung": "T1-85", "fold": 0, "seed": 0, "epoch": 3}

    template = (Path(__file__).resolve().parents[3] / "submissions" / "templates" /
                "task1-resnet18-fpn" / "inference.py")
    spec = importlib.util.spec_from_file_location("task1_resnet_submission", template)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = module.load_model(checkpoint)
    result = module.inference(0, None, np.zeros((128, 128, 3), dtype=np.uint8), bundle)
    assert list(result) == ["keypoints", "tool_tissue_distance"]
    assert len(result["keypoints"]) == 2
    assert all(np.isfinite(result["keypoints"]))
    assert isinstance(result["tool_tissue_distance"], float)
