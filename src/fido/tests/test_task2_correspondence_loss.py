import torch
from torch.utils.data import DataLoader, Dataset

from fido.losses.task2_contrastive import dense_infonce, sample_intraimage_negatives
from fido.losses.task2_contrastive import sample_positive_pairs
from fido.models.task2_common import FundusDenseEncoder, OctDenseEncoder, Task2CommonModel
from fido.train.train_task2_common import _DerangedValidation, select_group_fold


def test_encoders_are_separate_and_l2_normalized():
    model = Task2CommonModel(descriptor_dim=16)
    assert isinstance(model.fundus_encoder, FundusDenseEncoder)
    assert isinstance(model.oct_encoder, OctDenseEncoder)
    assert not set(map(id, model.fundus_encoder.parameters())) & set(map(id, model.oct_encoder.parameters()))
    output = model(torch.rand(2, 3, 64, 64), torch.rand(2, 1, 32, 48))
    assert torch.allclose(output["fundus_desc"].norm(dim=1), torch.ones(2, 8, 8), atol=1e-5)
    assert torch.allclose(output["oct_desc"].norm(dim=1), torch.ones(2, 4, 6), atol=1e-5)


def test_positive_sampler_recovers_known_translation():
    fundus = torch.randn(1, 4, 81, 81)
    oct_desc = torch.randn(1, 4, 5, 5)
    # OCT canonical [0,1] maps to a 41x41 square whose top-left is (32,48).
    gt = torch.tensor([[[40.0, 0.0, 32.0], [0.0, 40.0, 48.0], [0.0, 0.0, 1.0]]])
    pairs = sample_positive_pairs(fundus, oct_desc, gt, torch.ones(1, 1, 81, 81, dtype=torch.bool),
                                  k=25)
    oct_in_source_pixels = pairs.oct_xy * 10.0
    assert torch.allclose(pairs.fundus_xy - oct_in_source_pixels,
                          torch.tensor([32.0, 48.0]), atol=1e-5)


def test_sampler_excludes_invalid_fundus_support():
    valid = torch.ones(1, 1, 9, 9, dtype=torch.bool)
    valid[:, :, :5] = False
    gt = torch.tensor([[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 1.0]]])
    pairs = sample_positive_pairs(torch.randn(1, 8, 9, 9), torch.randn(1, 8, 3, 3), gt,
                                  valid, k=9)
    assert (pairs.fundus_xy[:, 1] >= 5).all()


def test_infonce_prefers_true_positive():
    negatives = torch.tensor([[0.0, -0.5], [0.0, -0.5]])
    assert dense_infonce(torch.tensor([1.0, 1.0]), negatives, 0.1) < dense_infonce(
        torch.tensor([-1.0, -1.0]), negatives, 0.1)


def test_negatives_respect_exclusion_radius():
    fundus = torch.zeros(1, 2, 5, 5)
    fundus[0, 0] = torch.arange(25).reshape(5, 5)
    oct_desc = torch.ones(1, 2, 1, 1)
    gt = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]])
    valid = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    pairs = sample_positive_pairs(fundus, oct_desc, gt, valid, k=1)
    negatives = sample_intraimage_negatives(pairs, fundus, valid, count=8,
                                             exclusion_radius_px=2.0)
    assert negatives.shape == (1, 8)
    assert not (negatives == 12.0).any()


def test_select_group_fold_uses_requested_fold_without_group_leakage():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    train_0, val_0 = select_group_fold(groups, n_folds=4, fold=0, seed=17)
    train_2, val_2 = select_group_fold(groups, n_folds=4, fold=2, seed=17)
    assert set(val_0) != set(val_2)
    assert set(torch.tensor(train_2).tolist()) | set(torch.tensor(val_2).tolist()) == set(range(8))
    assert not (set(groups[i] for i in train_2) & set(groups[i] for i in val_2))


class _Cases(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"enface": torch.tensor([float(index)]), "scenario": "s",
                "frame_id": str(index)}


def test_global_oct_derangement_has_no_autopair_in_last_singleton_batch():
    loader = DataLoader(_DerangedValidation(_Cases()), batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches[-1]["case_id"]) == 1
    pairs = [(target, source) for batch in batches
             for target, source in zip(batch["case_id"], batch["shuffle_source_id"])]
    assert len(pairs) == 5
    assert all(target != source for target, source in pairs)
    assert {source for _, source in pairs} == {f"s/{i}" for i in range(5)}
