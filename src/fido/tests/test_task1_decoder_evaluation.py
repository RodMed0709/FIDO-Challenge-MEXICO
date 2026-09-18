import numpy as np
import pytest

from analysis.evaluate_task1_decoders import load_cache, summarize_decoder


def _cache_arrays():
    logits = np.full((2, 1, 4, 4), -20.0, dtype=np.float32)
    logits[0, 0, 1, 2] = 20.0
    logits[1, 0, 3, 0] = 20.0
    return {
        "logits": logits,
        "ground_truth": np.asarray([[512.0, 256.0], [0.0, 768.0]], dtype=np.float32),
        "scenarios": np.asarray(["A", "B"]),
        "case_ids": np.asarray(["A/1", "B/1"]),
        "image_hw": np.asarray([1024, 1024]),
        "metadata_json": np.asarray('{"fold": 0}'),
    }


def test_summary_reports_official_metrics_and_scenarios():
    result = summarize_decoder(_cache_arrays(), "dark", temperature=1.0, window=3)
    assert result["n"] == 2
    assert result["auc"] == 1.0
    assert result["pck"]["1"] == 1.0
    assert set(result["per_scenario"]) == {"A", "B"}
    assert len(result["cdf"]) == 11


def test_cache_rejects_duplicate_case_ids(tmp_path):
    arrays = _cache_arrays()
    arrays["case_ids"] = np.asarray(["same", "same"])
    path = tmp_path / "cache.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="unique"):
        load_cache(path)
