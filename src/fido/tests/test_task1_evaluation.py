from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from fido.data import task1 as task1_data
from fido.data.task1 import Task1Dataset, first_task1_split
from fido.eval_task1 import evaluate_distances, evaluate_keypoints, write_evaluation_outputs
from fido.train.train_task1_unet import (
    fit_distance_calibration_samples,
    load_distance_calibration,
    save_distance_calibration,
)


def _fake_cases(n_scenarios: int = 10) -> list[dict]:
    return [
        {"scenario": f"Scenario_{scenario:02d}", "frame_id": f"{frame:04d}"}
        for scenario in range(n_scenarios)
        for frame in range(3)
    ]


def test_distance_auc_keeps_unmeasurable_cases():
    out = evaluate_distances(
        pred=np.array([10.0, 159.2]),
        gt=np.array([10.0, 300.0]),
        scenarios=np.array(["S1", "S2"]),
        measured_mask=np.array([True, False]),
    )

    assert out["n"] == 2
    assert out["coverage"] == 0.5
    assert out["predictions"].tolist() == [10.0, 159.2]
    assert out["cdf"].shape == (11,)


def test_group_split_has_no_scenario_overlap():
    cases = _fake_cases()
    train, val = first_task1_split(cases)

    assert set(train.scenario).isdisjoint(set(val.scenario))
    assert len(train) + len(val) == len(cases)
    assert set(train.case_id) | set(val.case_id) == {
        f"{case['scenario']}/{case['frame_id']}" for case in cases
    }


def test_group_split_rejects_duplicate_case_ids():
    cases = _fake_cases()
    with pytest.raises(ValueError, match="Duplicate Task 1 case IDs"):
        first_task1_split(cases + [cases[0].copy()])


def test_keypoint_metrics_report_pooled_and_per_scenario():
    out = evaluate_keypoints(
        pred_xy=np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]]),
        gt_xy=np.zeros((3, 2)),
        scenarios=np.array(["S1", "S1", "S2"]),
    )

    assert out["n"] == 3
    assert out["errors"].tolist() == [0.0, 5.0, 10.0]
    assert set(out["per_scenario"]) == {"S1", "S2"}
    assert out["per_scenario"]["S1"]["n"] == 2


def test_evaluation_writes_one_row_per_case_and_metrics(tmp_path):
    out = evaluate_distances(
        [1.0, 2.0], [1.0, 4.0], ["S1", "S2"], [True, False]
    )
    cases_path, metrics_path = write_evaluation_outputs(
        tmp_path / "predictions.jsonl", out, ["S1/1", "S2/2"]
    )

    assert len(cases_path.read_text(encoding="utf-8").splitlines()) == 2
    assert '"coverage": 0.5' in metrics_path.read_text(encoding="utf-8")


def test_evaluation_export_rejects_duplicate_case_ids(tmp_path):
    out = evaluate_keypoints([[0, 0], [1, 1]], [[0, 0], [0, 0]], ["S1", "S1"])
    with pytest.raises(ValueError, match="unique"):
        write_evaluation_outputs(tmp_path / "predictions.csv", out, ["S1/1", "S1/1"])


def test_distance_calibration_uses_only_train_case_ids():
    calibration = fit_distance_calibration_samples(
        [0.0, 10.0, np.nan], [2.0, 12.0, 100.0], ["S1/1", "S1/2", "S2/1"]
    )
    assert calibration.scale == pytest.approx(1.0)
    assert calibration.offset == pytest.approx(2.0)
    assert calibration.fallback == 12.0  # mediana de TODOS los GT train: 2,12,100
    assert set(calibration.fit_case_ids).isdisjoint({"S9/val-extreme"})


def test_calibration_provenance_excludes_actual_validation_split():
    train, val = first_task1_split(_fake_cases())
    calibration = fit_distance_calibration_samples(
        np.arange(len(train), dtype=float),
        np.arange(len(train), dtype=float) + 1.0,
        train.case_id,
    )
    assert set(calibration.fit_case_ids) == set(train.case_id)
    assert set(calibration.fit_case_ids).isdisjoint(set(val.case_id))


def test_distance_calibration_round_trip(tmp_path):
    calibration = fit_distance_calibration_samples(
        [0.0, 10.0, np.nan], [2.0, 12.0, 100.0], ["S1/1", "S1/2", "S2/1"]
    )
    path = tmp_path / "distance_calibration.json"
    save_distance_calibration(path, calibration)
    assert load_distance_calibration(path) == calibration


def test_discovery_io_failure_retries_same_resource_and_fails_contextually(monkeypatch, tmp_path):
    path = tmp_path / "Scenario_03" / "Numerical" / "0042.json"
    calls = []

    def broken(candidate):
        calls.append(candidate)
        raise OSError("network")

    monkeypatch.setattr(task1_data, "_cannula_is_active", broken)
    with pytest.raises(OSError, match="Scenario_03/0042"):
        task1_data._active_or_none(path, max_retries=2)
    assert calls == [path, path, path]


def test_scenario_listing_failure_retries_and_never_silently_skips():
    class BrokenNumerical:
        parent = SimpleNamespace(name="Scenario_07")

        def __init__(self):
            self.calls = 0

        def is_dir(self):
            return True

        def glob(self, pattern):
            assert pattern == "*.json"
            self.calls += 1
            raise OSError("listing unavailable")

    numerical = BrokenNumerical()
    with pytest.raises(OSError, match="Scenario_07/Numerical"):
        task1_data._list_scenario_jsons(numerical, max_retries=2)
    assert numerical.calls == 3


def test_root_listing_failure_retries_and_fails_contextually():
    class BrokenRoot:
        def __init__(self):
            self.calls = 0

        def glob(self, pattern):
            assert pattern == "Scenario_*"
            self.calls += 1
            raise OSError("root listing unavailable")

        def __str__(self):
            return "/broken/task1"

    root = BrokenRoot()
    with pytest.raises(OSError, match="Failed to list Task 1 root"):
        task1_data._list_scenario_dirs(root, max_retries=2)
    assert root.calls == 3


def test_dataset_retries_same_case_without_crossing_subset():
    dataset = object.__new__(Task1Dataset)
    dataset.cases = [{"scenario": "S1", "frame_id": "0001"},
                     {"scenario": "S2", "frame_id": "0002"}]
    dataset.max_load_retries = 2
    seen = []

    def flaky(case):
        seen.append(case["scenario"])
        if len(seen) < 3:
            raise OSError("transient")
        return {"scenario": case["scenario"]}

    dataset._load_case = flaky
    assert dataset[0]["scenario"] == "S1"
    assert seen == ["S1", "S1", "S1"]


def test_dataset_failure_names_original_case():
    dataset = object.__new__(Task1Dataset)
    dataset.cases = [{"scenario": "S1", "frame_id": "0001"}]
    dataset.max_load_retries = 1
    dataset._load_case = lambda case: (_ for _ in ()).throw(OSError("broken"))

    with pytest.raises(OSError, match="S1/0001"):
        dataset[0]
