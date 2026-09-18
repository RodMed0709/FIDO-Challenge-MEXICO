import numpy as np
import pytest

from analysis.measure_task1_distance_ceiling import (
    _oracle_gap,
    checkpoint_metadata,
    ground_truth_distance,
    mean_finite,
    validate_calibration_provenance,
)
from fido.eval_task1 import evaluate_distances
from fido.train.train_task1_unet import (
    DistanceCalibration,
    load_distance_calibration,
    save_distance_calibration,
)


def test_mean_finite_uses_both_orthogonal_measurements():
    assert mean_finite([10.0, 14.0]) == 12.0
    assert mean_finite([np.nan, 14.0]) == 14.0
    assert np.isnan(mean_finite([np.nan, np.nan]))


def test_oracle_gap_matches_training_geometry():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[6, 10] = 11
    mask[15, 8:14] = 1
    assert _oracle_gap(mask) == 9.0


def test_oracle_gap_is_unmeasured_without_required_classes():
    assert np.isnan(_oracle_gap(np.zeros((8, 8), dtype=np.uint8)))
    assert np.isnan(_oracle_gap(None))


def test_ground_truth_distance_applies_official_divide_by_ten():
    annotation = {"Ground Truth": {"Task 1": [100, 200, 1234]}}
    assert ground_truth_distance(annotation) == 123.4


def test_all_case_denominator_keeps_oracle_misses():
    metrics = evaluate_distances(
        [10.0, 999.0, 30.0], [10.0, 20.0, 30.0], ["S1", "S1", "S2"],
        [True, False, True], fallback=20.0,
    )
    assert metrics["n"] == 3
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["predictions"].tolist() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize(
    "fit_ids, message",
    [
        (("A",), "missing=1"),
        (("A", "B", "X"), "extra=1"),
        (("A", "B", "V"), "val_overlap=1"),
        (("A", "A", "B"), "duplicates"),
    ],
)
def test_calibration_provenance_rejects_missing_extra_leak_and_duplicates(fit_ids, message):
    with pytest.raises(RuntimeError, match=message):
        validate_calibration_provenance(fit_ids, ("A", "B"), ("V",))


def test_calibration_provenance_accepts_exact_train_set_in_any_order():
    validate_calibration_provenance(("B", "A"), ("A", "B"), ("V",))


def test_calibration_sidecar_load_preserves_provenance(tmp_path):
    expected = DistanceCalibration(0.7, 2.0, 99.0, ("S1/1", "S1/2"))
    path = tmp_path / "distance_calibration.json"
    save_distance_calibration(path, expected)
    assert load_distance_calibration(path) == expected


def test_checkpoint_metadata_only_keeps_linkage_fields():
    raw = {"state_dict": {}, "fold": 0, "seed": 7, "epoch": 2, "optimizer": {}}
    assert checkpoint_metadata(raw) == {"fold": 0, "seed": 7, "epoch": 2}
    assert checkpoint_metadata({"encoders.0.weight": object()}) == {}
