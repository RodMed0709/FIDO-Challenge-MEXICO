from __future__ import annotations

import torch


def configure_dino_stage(model, stage: str, last_blocks: int = 4) -> None:
    if stage not in {"lp", "partial_ft"}:
        raise ValueError("stage must be lp or partial_ft; full FT is intentionally unsupported")
    for branch in (model.fundus, model.oct):
        branch.training_stage = stage
        branch.requires_grad_(True)
        branch.backbone.requires_grad_(False)
        if stage == "partial_ft":
            for block in branch.backbone.blocks[-last_blocks:]:
                block.requires_grad_(True)


def dino_param_groups(model, stage: str, head_lr: float = 1e-4,
                      last_block_lr: float = 1e-5, layer_decay: float = 0.75,
                      weight_decay: float = 0.05) -> list[dict]:
    configure_dino_stage(model, stage)
    groups = []
    seen: set[int] = set()

    def add(name, parameters, lr):
        values = [parameter for parameter in parameters if parameter.requires_grad]
        if not values:
            return
        identifiers = {id(parameter) for parameter in values}
        if seen & identifiers:
            raise RuntimeError("parameter appears in multiple optimizer groups")
        seen.update(identifiers)
        groups.append({"name": name, "params": values, "lr": lr,
                       "weight_decay": weight_decay})

    for branch_name, branch in (("fundus", model.fundus), ("oct", model.oct)):
        add(f"{branch_name}_heads",
            list(branch.input_adapter.parameters()) + list(branch.laterals.parameters())
            + list(branch.refine.parameters()), head_lr)
        if stage == "partial_ft":
            blocks = list(branch.backbone.blocks)
            for depth, block in enumerate(reversed(blocks[-4:])):
                add(f"{branch_name}_block_{len(blocks)-1-depth}", block.parameters(),
                    last_block_lr * (layer_decay ** depth))
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError("some trainable DINO parameters are missing from optimizer groups")
    return groups


def build_dino_optimizer(model, stage: str, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(dino_param_groups(model, stage, **kwargs))
