import json
from pathlib import Path

import numpy as np
import pytest

from analysis.diagnose_task2_error import require_task2_training_root as diagnose_root_guard
from analysis.measure_task2_correspondence import (
    build_oct_derangement,
    require_task2_training_root as correspondence_root_guard,
)
from fido.data.task2 import Task2Dataset, _retry_discovery
from fido.eval_task2 import evaluate_task2_cases, substitute_gt_components, write_case_jsonl
from fido.geometry import compose_similarity, corner_auc, corner_error


def _matrix(tx, ty, theta, scale):
    return compose_similarity(
        tx, ty, np.cos(theta), np.sin(theta), scale, reflect=True,
    )


def test_task2_case_evaluator_matches_geometry_auc():
    gt = np.stack([_matrix(100, 200, 0.2, 160), _matrix(300, 400, -0.4, 150)])
    pred = gt.copy()
    pred[0, 0, 2] += 3.0
    pred[1, 1, 2] += 7.0

    out = evaluate_task2_cases(pred, gt, np.array(["S1", "S2"]))
    errors = corner_error(pred, gt)

    assert out["auc"] == corner_auc(errors)
    assert np.allclose(out["errors"], errors)
    assert out["n_cases"] == 2
    assert set(out["per_scenario"]) == {"S1", "S2"}


def test_task2_case_evaluator_rejects_mismatched_lengths():
    matrix = np.eye(3, dtype=np.float64)[None]
    with np.testing.assert_raises_regex(ValueError, "same number of cases"):
        evaluate_task2_cases(matrix, matrix, np.array(["S1", "S2"]))


def test_component_substitution_changes_only_requested_component():
    gt = np.stack([_matrix(100, 200, 0.5, 170), _matrix(300, 400, -0.2, 140)])
    pred = np.stack([_matrix(10, 20, -1.0, 90), _matrix(30, 40, 1.2, 210)])

    position = substitute_gt_components(pred, gt, ("position",))
    rotation = substitute_gt_components(pred, gt, ("rotation",))
    scale = substitute_gt_components(pred, gt, ("scale",))

    assert np.allclose(position[:, :2, 2], gt[:, :2, 2])
    assert np.allclose(rotation[:, :2, 2], pred[:, :2, 2])
    assert np.allclose(scale[:, :2, 2], pred[:, :2, 2])
    assert np.allclose(substitute_gt_components(pred, gt, ("position", "rotation", "scale")), gt)


def test_case_jsonl_has_one_valid_row_per_case(tmp_path):
    gt = np.stack([_matrix(100, 200, 0.2, 160), _matrix(300, 400, -0.4, 150)])
    pred = gt.copy()
    result = evaluate_task2_cases(pred, gt, np.array(["S1", "S2"]), case_ids=["a", "b"])
    path = tmp_path / "cases.jsonl"

    write_case_jsonl(path, result["cases"])

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["case_id"] for row in rows] == ["a", "b"]
    assert all(row["scenario"] in {"S1", "S2"} for row in rows)
    assert all(isinstance(row["corner_error"], float) for row in rows)


def test_case_jsonl_rejects_duplicate_case_within_condition(tmp_path):
    rows = [{"case_id": "a", "condition": "paired"},
            {"case_id": "a", "condition": "paired"}]
    with pytest.raises(ValueError, match="unique case_id"):
        write_case_jsonl(tmp_path / "duplicate.jsonl", rows)


def test_task2_dataset_never_substitutes_case_after_io_failure(monkeypatch):
    dataset = Task2Dataset.__new__(Task2Dataset)
    dataset.cases = [
        {"scenario": "S1", "frame_id": "a"},
        {"scenario": "S2", "frame_id": "b"},
    ]
    loaded = []

    def fail(case):
        loaded.append(case["frame_id"])
        raise OSError("broken")

    monkeypatch.setattr(dataset, "_load_case", fail)
    with pytest.raises(OSError, match="S1/a"):
        dataset[0]
    assert loaded == ["a"]


def test_discovery_retries_same_resource_then_fails(monkeypatch):
    monkeypatch.setattr("fido.data.task2.time.sleep", lambda _: None)
    calls = []

    def fail():
        calls.append("same")
        raise OSError("transient")

    with pytest.raises(OSError, match="Scenario_03/00042"):
        _retry_discovery(fail, "Scenario_03/00042", attempts=3)
    assert calls == ["same", "same", "same"]


def test_oct_derangement_preserves_fold_and_paired_alignment():
    paired = np.array([7, 11, 19, 23])
    target_order = paired.copy()
    shuffled = build_oct_derangement(paired)
    assert np.array_equal(paired, target_order)
    assert set(shuffled.tolist()) == set(paired.tolist())
    assert np.all(shuffled != paired)


def test_mock_is_rejected_for_per_case_gate():
    for mock_name in ("Mock Test", "Mock_Test", "Mock-Test"):
        mock_root = Path("data") / mock_name / "Task 2"
        for guard in (diagnose_root_guard, correspondence_root_guard):
            with pytest.raises(RuntimeError, match="Mock Test labels are forbidden"):
                guard(mock_root)
