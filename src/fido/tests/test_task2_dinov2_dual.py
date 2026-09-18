import hashlib
import json
import random

import numpy as np
import pytest
import torch

from fido.data.task2_transforms import ScaleAugmentationConfig
from fido.models.task2_dinov2 import DualDinoCommonEncoder
from fido.train.task2_dinov2_optim import configure_dino_stage, dino_param_groups
from fido.train.train_task2_common import (
    ValidMicrobatchAccumulator,
    capture_rng_state,
    attach_resume_state,
    atomic_torch_save,
    common_checkpoint_payload,
    restore_rng_state,
    validate_dino_prerequisites,
    validate_resume_payload,
)


def _model():
    return DualDinoCommonEncoder(descriptor_dim=16, pretrained=False,
                                 fundus_input_size=56, oct_input_size=56)


def test_dual_dino_does_not_share_weights_and_has_stride_seven_descriptors():
    model = _model()
    assert (model.fundus.backbone.blocks[0].attn.qkv.weight.data_ptr()
            != model.oct.backbone.blocks[0].attn.qkv.weight.data_ptr())
    output = model(torch.rand(1, 3, 64, 64), torch.rand(1, 1, 32, 48))
    assert output["fundus_desc"].shape == (1, 16, 8, 8)
    assert output["oct_desc"].shape == (1, 16, 8, 8)
    assert torch.allclose(output["fundus_desc"].norm(dim=1), torch.ones(1, 8, 8), atol=1e-5)


def test_lp_freezes_both_backbones_but_trains_adapters_and_fpn():
    model = _model()
    configure_dino_stage(model, "lp")
    model.train()
    assert not any(parameter.requires_grad for parameter in model.fundus.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.oct.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.fundus.input_adapter.parameters())
    assert not model.fundus.backbone.training
    groups = dino_param_groups(model, "lp")
    assert {group["name"] for group in groups} == {"fundus_heads", "oct_heads"}


def test_partial_ft_unfreezes_only_last_four_blocks_with_llrd():
    model = _model()
    groups = dino_param_groups(model, "partial_ft", head_lr=1e-4,
                               last_block_lr=1e-5, layer_decay=.75)
    model.train()
    for branch in (model.fundus, model.oct):
        assert not any(parameter.requires_grad for block in branch.backbone.blocks[:-4]
                       for parameter in block.parameters())
        assert all(parameter.requires_grad for block in branch.backbone.blocks[-4:]
                   for parameter in block.parameters())
        assert all(block.training for block in branch.backbone.blocks[-4:])
        assert not any(block.training for block in branch.backbone.blocks[:-4])
    by_name = {group["name"]: group["lr"] for group in groups}
    assert by_name["fundus_heads"] == 1e-4
    assert by_name["fundus_block_11"] == 1e-5
    assert by_name["fundus_block_8"] == 1e-5 * .75**3
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)


def test_checkpoint_records_coordinate_frame_stage_and_optimizer_provenance():
    model = _model()
    config = ScaleAugmentationConfig(.8, 1.2, 4.8, 5.2, (0, 1))
    payload = common_checkpoint_payload(
        model, epoch=13, seed=0, fold=0, stats={"source_indices": [0, 1]},
        scale_config=config, encoder="dinov2", phase="partial_ft",
        optimizer_provenance={"head_lr": 1e-4, "last_block_lr": 1e-5,
                              "layer_decay": .75, "amp_dtype": "bfloat16"})
    assert payload["coordinate_frame"] == "task2_canonical_v1"
    assert payload["encoder"] == "dinov2"
    assert payload["phase"] == "partial_ft"
    assert payload["optimizer_provenance"]["amp_dtype"] == "bfloat16"


def test_valid_accumulation_skips_invalid_and_averages_short_remainder_exactly():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    accumulator = ValidMicrobatchAccumulator(optimizer, [parameter], target_count=3)
    # An invalid microbatch calls neither backward nor the counter.
    accumulator.backward((parameter - 1.2).square())
    accumulator.backward((parameter - 1.4).square())
    assert not accumulator.ready()
    assert accumulator.valid_count == 2
    accumulator.step()
    # Mean gradient at p=1 is mean(-0.4,-0.8)=-0.6; SGD gives p=1.6.
    assert torch.allclose(parameter, torch.tensor(1.6))
    assert accumulator.valid_count == 0


def test_rng_capture_restore_includes_python_numpy_torch_and_sampler():
    generator = torch.Generator().manual_seed(91)
    random.seed(7); np.random.seed(8); torch.manual_seed(9)
    state = capture_rng_state(generator)
    expected = (random.random(), np.random.rand(), torch.rand(()), torch.rand((), generator=generator))
    restore_rng_state(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(()), torch.rand((), generator=generator))
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2]) and torch.equal(actual[3], expected[3])


def test_atomic_resume_checkpoint_round_trips_optimizer_scheduler_and_rng(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(3)
    (parameter.square()).backward(); optimizer.step(); scheduler.step()
    payload = attach_resume_state(
        {"model": {"p": parameter.detach().clone()}, "phase": "lp", "epoch": 1},
        global_step=1, phase_epoch=1, optimizer=optimizer, scheduler=scheduler,
        generator=generator, best_phase_score=.25, phase_metrics={"lp": {"m1": .1}})
    path = tmp_path / "resume.pth"
    atomic_torch_save(payload, path)
    assert path.exists() and not (tmp_path / "resume.pth.tmp").exists()
    loaded = torch.load(path, weights_only=False)
    assert loaded["global_step"] == 1 and loaded["phase_epoch"] == 1
    assert loaded["optimizer"]["state"] and loaded["scheduler"]["last_epoch"] == 1
    assert loaded["best_phase_score"] == .25 and "sampler_generator" in loaded["rng_state"]


def test_prerequisites_require_exact_fold_ids_schema_and_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "model_1.pth"
    train_hash, validation_hash = "1" * 64, "2" * 64
    checkpoint_payload = {"coordinate_frame": "task2_canonical_v1",
                          "gate_schema": "task2_common_representation_v1",
                          "encoder": "cnn", "fold": 0, "seed": 0,
                          "validation_indices": [4, 5],
                          "train_case_ids_sha256": train_hash,
                          "validation_case_ids_sha256": validation_hash,
                          "recipe": {"schema": "task2_common_recipe_v1"}}
    torch.save(checkpoint_payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    validation = [4, 5]
    gate = {"gate_pass": True, "selection_uses_mock": False,
            "expected_convention": "transpose__flip_u__flip_v",
            "canonical_to_native_matrix": [[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            "gate_schema": "task2_enface_convention_gate_v1",
            "train_case_ids_sha256": train_hash,
            "validation_case_ids_sha256": validation_hash,
            "fold": 0, "seed": 0, "validation_indices": validation}
    control = {"gate_pass": True, "selection_uses_mock": False, "fold": 0, "seed": 0,
               "validation_indices": validation, "coordinate_frame": "task2_canonical_v1",
               "checkpoint_sha256": digest, "metrics": {},
               "gate_schema": "task2_common_representation_v1",
               "train_case_ids_sha256": train_hash,
               "validation_case_ids_sha256": validation_hash}
    gate_path, control_path = tmp_path / "gate.json", tmp_path / "control.json"
    gate_path.write_text(json.dumps(gate)); control_path.write_text(json.dumps(control))
    loaded_gate, loaded_control = validate_dino_prerequisites(
        gate_path, control_path, checkpoint, fold=0, seed=0, validation_indices=validation,
        train_ids_sha256=train_hash, validation_ids_sha256=validation_hash)
    assert len(loaded_gate["artifact_sha256"]) == 64
    assert len(loaded_control["artifact_sha256"]) == 64
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        validate_dino_prerequisites(gate_path, control_path, checkpoint,
                                    fold=0, seed=0, validation_indices=validation,
                                    train_ids_sha256=train_hash,
                                    validation_ids_sha256=validation_hash)


@pytest.mark.parametrize("field,bad", [
    ("head_lr", 2e-4), ("layer_decay", .5), ("grad_accum", 8),
    ("amp_dtype", "float16"), ("batch_size", 2), ("fundus_input_size", 518),
    ("lp_epochs", 4), ("k", 32), ("temperature", .2),
    ("scheduler", "other"),
])
def test_resume_fails_closed_on_every_recipe_tamper(field, bad):
    recipe = {"schema": "task2_common_recipe_v1", "head_lr": 1e-4,
              "layer_decay": .75, "grad_accum": 4, "amp_dtype": "bfloat16",
              "batch_size": 1, "fundus_input_size": 1022, "lp_epochs": 3,
              "k": 64, "temperature": .1, "scheduler": "warmup_5pct_cosine_v1"}
    payload = {"coordinate_frame": "task2_canonical_v1", "encoder": "dinov2",
               "fold": 0, "seed": 0, "gate_schema": "task2_common_representation_v1",
               "recipe": dict(recipe)}
    payload["recipe"][field] = bad
    with pytest.raises(ValueError, match="recipe mismatch"):
        validate_resume_payload(payload, recipe, encoder="dinov2", fold=0, seed=0)


@pytest.mark.parametrize("tamper,match", [
    ("matrix", "authoritative C"), ("train_ids", "ID hashes"),
    ("gate_schema", "gate schema"), ("checkpoint_frame", "checkpoint provenance"),
])
def test_prerequisites_fail_closed_for_provenance_tampering(tmp_path, tamper, match):
    train_hash, val_hash, validation = "1" * 64, "2" * 64, [4, 5]
    checkpoint_payload = {"coordinate_frame": "task2_canonical_v1",
        "gate_schema": "task2_common_representation_v1", "encoder": "cnn",
        "fold": 0, "seed": 0, "validation_indices": validation,
        "train_case_ids_sha256": train_hash, "validation_case_ids_sha256": val_hash,
        "recipe": {"schema": "task2_common_recipe_v1"}}
    if tamper == "checkpoint_frame":
        checkpoint_payload["coordinate_frame"] = "native"
    checkpoint = tmp_path / "model.pth"; torch.save(checkpoint_payload, checkpoint)
    gate = {"gate_pass": True, "selection_uses_mock": False,
        "train_case_ids_sha256": train_hash, "validation_case_ids_sha256": val_hash,
        "expected_convention": "transpose__flip_u__flip_v",
        "canonical_to_native_matrix": [[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        "gate_schema": "task2_enface_convention_gate_v1", "fold": 0, "seed": 0,
        "validation_indices": validation}
    if tamper == "matrix": gate["canonical_to_native_matrix"][0][0] = 1.0
    if tamper == "train_ids": gate["train_case_ids_sha256"] = "3" * 64
    if tamper == "gate_schema": gate["gate_schema"] = "v0"
    control = {"gate_pass": True, "selection_uses_mock": False, "fold": 0, "seed": 0,
        "validation_indices": validation, "coordinate_frame": "task2_canonical_v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "metrics": {},
        "gate_schema": "task2_common_representation_v1",
        "train_case_ids_sha256": train_hash, "validation_case_ids_sha256": val_hash}
    gate_path, control_path = tmp_path / "gate.json", tmp_path / "control.json"
    gate_path.write_text(json.dumps(gate)); control_path.write_text(json.dumps(control))
    with pytest.raises(ValueError, match=match):
        validate_dino_prerequisites(gate_path, control_path, checkpoint, fold=0, seed=0,
            validation_indices=validation, train_ids_sha256=train_hash,
            validation_ids_sha256=val_hash)
