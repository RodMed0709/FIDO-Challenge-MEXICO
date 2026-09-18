#!/usr/bin/env python3
"""Score a FIDO submission locally, using the organisers' own code.

This is a thin wrapper around `vendor/fido/Codabench Bundle/`. The ingestion and
scoring modules are imported *unmodified* — only their hardcoded `/app/...` data
directory is redirected at runtime, exactly as the organisers' own run_local.py
does. Anything this reports is what Codabench will report, minus the hidden test
set.

Why a wrapper instead of the vendored run_local.py: that file hardcodes paths to
a machine none of us have, and editing vendored code destroys the guarantee that
what we score against is byte-identical to what the graders run.

Usage
-----
    # Both tasks against the Mock Test data
    python eval/run_local.py --submission submissions/t2-rung01

    # One task
    python eval/run_local.py --submission submissions/t2-rung01 --task registration

    # Point at different test data
    python eval/run_local.py --submission ... --data "D:/.../Mock Test"

    # Record the result into the submission folder for the record
    python eval/run_local.py --submission submissions/t2-rung01 --save
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = PROJECT_ROOT / "vendor" / "fido" / "Codabench Bundle"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Mock Test"

# Mirrors the TASKS table in the organisers' run_local.py. The `id` is what gets
# passed to inference() as task_id, and decides the weight filename.
TASKS = {
    "keypoints": {
        "ingestion": BUNDLE_ROOT / "ingestion_program" / "ingestion_keypoints.py",
        "scoring": BUNDLE_ROOT / "scoring_program" / "scoring_keypoints.py",
        "data_subdir": "Task 1",
        "id": 0,
    },
    "registration": {
        "ingestion": BUNDLE_ROOT / "ingestion_program" / "ingestion_registration.py",
        "scoring": BUNDLE_ROOT / "scoring_program" / "scoring_registration.py",
        "data_subdir": "Task 2",
        "id": 1,
    },
}


def load_module(path: Path, name: str):
    if not path.exists():
        sys.exit(f"No encuentro el módulo del bundle: {path}\n"
                 f"¿Está vendorizado el repo oficial en {BUNDLE_ROOT.parent}?")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_module_main(module, argv: list[str]):
    """Call a bundle module's main() with a synthetic argv, restoring the real one."""
    saved = sys.argv[:]
    sys.argv = argv
    try:
        module.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = saved


def run_task(task_name: str, submission_dir: Path, data_root: Path) -> dict:
    config = TASKS[task_name]
    data_dir = data_root / config["data_subdir"]

    if not data_dir.is_dir():
        return {"final_score": 0.0, "error": f"No existe el directorio de datos: {data_dir}"}

    with tempfile.TemporaryDirectory(prefix=f"fido_{task_name}_") as tmp:
        predictions_dir = Path(tmp) / "predictions"
        scores_dir = Path(tmp) / "scores"
        predictions_dir.mkdir()
        scores_dir.mkdir()

        print(f"\n{'=' * 62}\n  INGESTION  ({task_name})\n{'=' * 62}")
        ingestion = load_module(config["ingestion"], f"ingestion_{task_name}")
        # The only mutation: redirect the container path to local data.
        ingestion.CHALLENGE_DATA_DIR = data_dir
        started = time.monotonic()
        run_module_main(ingestion, ["ingestion.py", str(predictions_dir), str(submission_dir)])
        wall_seconds = time.monotonic() - started

        predictions_path = predictions_dir / "predictions.json"
        if not predictions_path.exists():
            return {"final_score": 0.0, "error": "La ingestión no produjo predictions.json"}

        predictions = json.loads(predictions_path.read_text())
        print(f"\n  Predicciones escritas para {len(predictions)} caso(s)")

        print(f"\n{'=' * 62}\n  SCORING  ({task_name})\n{'=' * 62}")
        scoring = load_module(config["scoring"], f"scoring_{task_name}")
        scoring.CHALLENGE_DATA_DIR = data_dir
        run_module_main(scoring, ["scoring.py", str(predictions_dir), str(scores_dir)])

        scores = json.loads((scores_dir / "scores.json").read_text())
        scores["wall_seconds"] = round(wall_seconds, 2)

        durations_path = predictions_dir / "durations.json"
        if durations_path.exists():
            durations = json.loads(durations_path.read_text())
            per_case = durations.get("per_case_seconds", {})
            if per_case:
                values = sorted(per_case.values())
                scores["avg_case_seconds"] = round(sum(values) / len(values), 3)
                scores["max_case_seconds"] = round(values[-1], 3)
                # The graders drop any case over the limit to a score of zero, so
                # the worst case matters far more than the average.
                scores["per_case_limit"] = durations.get("per_case_time_limit_seconds")
            scores["timed_out_cases"] = durations.get("timed_out_cases", [])

        return scores


def check_required_files(submission_dir: Path, task_name: str) -> list[str]:
    """Replicate the ingestion program's own file check, before wasting a run.

    Note the weight filename is model_<task_id>.pth, NOT model.pth as the
    Codabench page says. Getting this wrong is a silent zero on the leaderboard.
    """
    task_id = TASKS[task_name]["id"]
    problems = []
    if not (submission_dir / "inference.py").is_file():
        problems.append("falta inference.py en la raíz de la submission")
    if not (submission_dir / f"model_{task_id}.pth").is_file():
        problems.append(f"falta model_{task_id}.pth en la raíz de la submission")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Evalúa una submission de FIDO localmente.")
    parser.add_argument("--submission", required=True, type=Path,
                        help="Carpeta con inference.py y model_<task_id>.pth")
    parser.add_argument("--task", choices=["keypoints", "registration", "both"], default="both")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"Raíz de los datos de prueba (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--save", action="store_true",
                        help="Escribir el resultado en <submission>/local_score.json")
    args = parser.parse_args()

    submission_dir = args.submission.resolve()
    if not submission_dir.is_dir():
        sys.exit(f"No existe la carpeta de submission: {submission_dir}")

    task_names = ["keypoints", "registration"] if args.task == "both" else [args.task]

    results = {}
    for task_name in task_names:
        problems = check_required_files(submission_dir, task_name)
        if problems:
            for problem in problems:
                print(f"ERROR [{task_name}]: {problem}")
            results[task_name] = {"final_score": 0.0, "error": "; ".join(problems)}
            continue
        try:
            results[task_name] = run_task(task_name, submission_dir, args.data)
        except Exception as exc:
            results[task_name] = {"final_score": 0.0, "error": f"{type(exc).__name__}: {exc}"}

    print(f"\n{'=' * 62}\n  RESUMEN\n{'=' * 62}")
    for task_name, scores in results.items():
        error = scores.get("error")
        if error:
            print(f"  {task_name:>13}:  FALLÓ — {error}")
            continue
        limit = scores.get("per_case_limit")
        worst = scores.get("max_case_seconds")
        timing = f"peor caso {worst}s" if worst is not None else "sin tiempos"
        if limit and worst:
            headroom = limit - worst
            timing += f" / límite {limit}s (margen {headroom:.1f}s)"
            if headroom < 0:
                timing += "  ¡EXCEDIDO!"
        print(f"  {task_name:>13}:  score = {scores['final_score']:.6f}"
              f"   casos = {scores.get('num_cases', '?')}"
              f"   timeouts = {len(scores.get('timed_out_cases') or [])}"
              f"   cuda = {bool(scores.get('cuda_available'))}")
        print(f"  {'':>13}   {timing}")
        if task_name == "keypoints":
            print(f"  {'':>13}   keypoint_auc = {scores.get('keypoint_auc')}"
                  f"   distance_auc = {scores.get('distance_auc')}")

    if args.save:
        payload = {
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(args.data),
            "results": results,
        }
        out = submission_dir / "local_score.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n  Guardado en {out}")

    if any(r.get("error") for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
