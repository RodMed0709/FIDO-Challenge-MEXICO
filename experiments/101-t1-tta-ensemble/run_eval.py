#!/usr/bin/env python3
"""T1-101: TTA + ensamble para el keypoint de Task 1, SIN reentrenar nada.

Que mide: sobre el fold-0 de validacion (2 escenarios que el checkpoint NUNCA
vio en entrenamiento -- GroupKFold por escenario, `fido.data.task1.task1_split`,
n_splits=5/fold=0/seed=0, el MISMO split que produjo `val_keypoint_auc=0.8539`
en T1-80/ATTACK_LADDER.md), compara `keypoint_auc` oficial (`corner_auc` sobre
umbrales enteros 0..10 px) para: baseline sin TTA, cada familia de TTA por
separado (flips+rotaciones de 90, zoom multi-escala), la combinacion, y el
ensamble de checkpoints (si hay mas de uno utilizable) con y sin TTA.

Datos: `data/Task 1/Scenario_XX.zip` (56 GB, NO se extraen -- el fundus
`Stereo Left/<frame>/microscope.png` se lee en memoria via `zipfile` + PIL,
igual que `analysis/measure_task1_segmentation_nan_fraction.py`).
`data/_annotations/Task 1/Scenario_XX/*.json` (extraido localmente, barato)
da el GT y el filtro de canula activa sin tocar los zips.

Eficiencia: cada VISTA (identity/hflip/vflip/rot90/180/270/zoom0.9/zoom1.1)
se corre UNA sola vez por (caso, checkpoint) y se cachea en memoria; las
distintas configuraciones (baseline, flip_rot, scale, combined) son
combinaciones lineales de esas 8 vistas ya calculadas, asi que no repiten
forward passes -- el costo total es O(N_casos x 8 vistas x N_checkpoints),
no O(N_casos x N_configs x N_vistas).

Uso:
    python experiments/101-t1-tta-ensemble/run_eval.py
    python experiments/101-t1-tta-ensemble/run_eval.py --n-per-scenario 100
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

torch.set_num_threads(8)

from fido.data.task1 import task1_split  # noqa: E402
from fido.eval_task1 import evaluate_keypoints  # noqa: E402
from fido.heatmap_decode import decode_heatmap  # noqa: E402
from fido.models.task1_keypoint import Task1KeypointModel  # noqa: E402
from fido.tta_keypoint import (  # noqa: E402
    default_flip_rotation_transforms,
    default_scale_transforms,
)

CANNULA_INACTIVE_SENTINEL = 1e6
FUNDUS_SIZE = 1024

# Los 3 archivos de pesos DISTINTOS de Task1KeypointModel que sobrevivieron
# localmente (hash sha256 + suma de un tensor, verificado a mano antes de
# escribir este script -- ver INFORME.md seccion "Checkpoints disponibles").
# `cnn_best` es el que reporta ATTACK_LADDER.md T1-80 (`val_keypoint_auc=0.8539`
# mejor epoca 5 / 0.8537 final epoca 20). Los otros dos son snapshots mas
# tempranos de entrenamientos anteriores (r01, r04..r06) -- se incluyen en el
# ensamble SOLO si la triage (ver `triage_checkpoints`) confirma que no son
# catastroficamente peores; promediar un checkpoint mucho peor con uno bueno
# puede empeorar el ensamble, no es un supuesto seguro por defecto.
CHECKPOINT_CANDIDATES = {
    "cnn_best_ep20": ROOT / "checkpoints_from_pod" / "latest" / "keypoint_0.8537.pth",
    "cnn_r01_early": ROOT / "checkpoints_from_pod" / "keypoint_only.pth",
    "cnn_r04_interim": ROOT / "artifacts" / "r04-interim" / "keypoint_only.pth",
}

VIEW_NAMES = ["identity", "hflip", "vflip", "rot90", "rot180", "rot270",
              "zoom_0.90", "zoom_1.10"]


def _all_views():
    return default_flip_rotation_transforms() + default_scale_transforms((0.9, 1.1))


def load_model(checkpoint_path: Path) -> Task1KeypointModel:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = Task1KeypointModel(base_channels=32, n_downsamples=4)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def cannula_active(annotation: dict) -> bool:
    ilm = annotation.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
    return ilm is not None and ilm < CANNULA_INACTIVE_SENTINEL


def build_case_cache(annotations_root: Path, cache_path: Path) -> list[dict]:
    """Replica exacta (mismo filtro, mismo orden) de
    `fido.data.task1.find_task1_cases` pero leyendo los JSON ya extraidos en
    `data/_annotations/` en vez de abrir los zips de 56 GB -- resultado
    identico porque son copias literales de `Scenario_XX/Numerical/*.json`."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cases = []
    for scenario_dir in sorted(annotations_root.glob("Scenario_*")):
        for json_path in sorted(scenario_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if cannula_active(data):
                cases.append({"scenario": scenario_dir.name, "frame_id": json_path.stem})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cases), encoding="utf-8")
    return cases


def load_ground_truth(annotations_root: Path, scenario: str, frame_id: str) -> np.ndarray:
    path = annotations_root / scenario / f"{frame_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["Ground Truth"]["Task 1"][:2], dtype=np.float64)


def load_fundus_tensor(zip_handle: zipfile.ZipFile, frame_id: str) -> torch.Tensor:
    name = f"Stereo Left/{frame_id}/microscope.png"
    with zip_handle.open(name) as handle:
        data = handle.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape[:2] != (FUNDUS_SIZE, FUNDUS_SIZE):
        raise ValueError(f"unexpected fundus shape {array.shape} for {frame_id}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def stratified_sample(val_cases: list[dict], n_per_scenario: int, seed: int) -> list[dict]:
    by_scenario: dict[str, list[dict]] = {}
    for case in val_cases:
        by_scenario.setdefault(case["scenario"], []).append(case)
    rng = np.random.default_rng(seed)
    sampled = []
    for scenario in sorted(by_scenario):
        pool = by_scenario[scenario]
        take = min(n_per_scenario, len(pool))
        indices = rng.choice(len(pool), size=take, replace=False)
        sampled.extend(pool[i] for i in sorted(indices.tolist()))
    return sampled


def compute_views_for_case(model: Task1KeypointModel, fundus: torch.Tensor,
                           views: list) -> dict[str, torch.Tensor]:
    """Un forward pass por vista; devuelve `{nombre: heatmap_logits}` YA
    des-transformado a la grilla canonica (identity)."""
    batch = fundus.unsqueeze(0)
    out = {}
    for view in views:
        transformed = view.transform_image(batch)
        with torch.no_grad():
            prediction = model(transformed, fundus_size=FUNDUS_SIZE)
        out[view.name] = view.untransform_heatmap(prediction["heatmap_logits"])
    return out


def decode_xy(mean_logits: torch.Tensor, temperature: float) -> np.ndarray:
    coords = decode_heatmap(mean_logits, mode="global", temperature=temperature)
    stride = FUNDUS_SIZE / mean_logits.shape[-1]
    return (coords[0, 0].numpy() * stride)


def triage_checkpoints(candidates: dict[str, Path], val_cases: list[dict],
                       annotations_root: Path, zip_root: Path,
                       n_per_scenario: int, seed: int) -> dict[str, dict]:
    """Evalua `keypoint_auc` (solo baseline `identity`, sin TTA) de cada
    checkpoint candidato sobre una muestra pequena, para decidir cuales
    entran al ensamble. No es la evaluacion final -- solo triage barato."""
    sample = stratified_sample(val_cases, n_per_scenario, seed)
    identity_view = default_flip_rotation_transforms()[:1]
    results = {}
    for name, path in candidates.items():
        if not path.exists():
            results[name] = {"available": False}
            continue
        model = load_model(path)
        preds, gts, scenarios = [], [], []
        cases_by_scenario: dict[str, list[dict]] = {}
        for case in sample:
            cases_by_scenario.setdefault(case["scenario"], []).append(case)
        for scenario, cases in cases_by_scenario.items():
            with zipfile.ZipFile(zip_root / f"{scenario}.zip") as zf:
                for case in cases:
                    fundus = load_fundus_tensor(zf, case["frame_id"])
                    views = compute_views_for_case(model, fundus, identity_view)
                    xy = decode_xy(views["identity"], float(model.heatmap_temperature))
                    preds.append(xy)
                    gts.append(load_ground_truth(annotations_root, scenario, case["frame_id"]))
                    scenarios.append(scenario)
        metrics = evaluate_keypoints(np.stack(preds), np.stack(gts), scenarios)
        results[name] = {"available": True, "n": metrics["n"], "auc": metrics["auc"],
                         "mean_error": metrics["mean_error"]}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path,
                        default=ROOT / "data" / "_annotations" / "Task 1")
    parser.add_argument("--zip-root", type=Path, default=ROOT / "data" / "Task 1")
    parser.add_argument("--case-cache", type=Path,
                        default=Path(__file__).parent / "cases_cache.json")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-per-scenario", type=int, default=100)
    parser.add_argument("--triage-n-per-scenario", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results.json")
    args = parser.parse_args()

    print("[1/5] Construyendo/leyendo la lista completa de casos (GT + filtro de canula)...")
    t0 = time.time()
    cases = build_case_cache(args.annotations_root, args.case_cache)
    print(f"      {len(cases)} casos activos en {time.time() - t0:.1f}s")

    print("[2/5] GroupKFold por escenario (mismo split que el entrenamiento del checkpoint)...")
    train_split, val_split = task1_split(cases, n_splits=args.n_splits,
                                         fold=args.fold, seed=args.split_seed)
    val_scenarios = sorted(set(val_split.scenario.tolist()))
    print(f"      train n={len(train_split)} ({sorted(set(train_split.scenario.tolist()))}), "
          f"val n={len(val_split)} ({val_scenarios})")
    val_cases = list(val_split.cases)

    print("[3/5] Triage de checkpoints candidatos (muestra pequena, solo baseline)...")
    triage = triage_checkpoints(CHECKPOINT_CANDIDATES, val_cases, args.annotations_root,
                                args.zip_root, args.triage_n_per_scenario, args.sample_seed)
    for name, info in triage.items():
        if info.get("available"):
            print(f"      {name}: keypoint_auc={info['auc']:.4f} (n={info['n']}, "
                  f"mean_error={info['mean_error']:.2f}px)")
        else:
            print(f"      {name}: NO DISPONIBLE en disco")

    usable = {name: path for name, path in CHECKPOINT_CANDIDATES.items()
             if triage.get(name, {}).get("available") and triage[name]["auc"] > 0.05}
    print(f"      checkpoints usables para ensamble: {list(usable)}")

    print(f"[4/5] Muestra principal: {args.n_per_scenario} casos/escenario "
          f"x {len(val_scenarios)} escenarios...")
    sample = stratified_sample(val_cases, args.n_per_scenario, args.sample_seed)
    print(f"      n={len(sample)}")

    all_views = _all_views()
    assert [v.name for v in all_views] == VIEW_NAMES

    per_checkpoint_views: dict[str, dict[str, list[np.ndarray]]] = {}
    per_checkpoint_time: dict[str, float] = {}
    ground_truth, scenarios = [], []

    for name, path in usable.items():
        print(f"      corriendo {name}...")
        model = load_model(path)
        temperature = float(model.heatmap_temperature)
        views_out: dict[str, list[np.ndarray]] = {v: [] for v in VIEW_NAMES}
        cases_by_scenario: dict[str, list[dict]] = {}
        for case in sample:
            cases_by_scenario.setdefault(case["scenario"], []).append(case)
        t_start = time.time()
        done = 0
        for scenario, cases in cases_by_scenario.items():
            with zipfile.ZipFile(args.zip_root / f"{scenario}.zip") as zf:
                for case in cases:
                    fundus = load_fundus_tensor(zf, case["frame_id"])
                    views = compute_views_for_case(model, fundus, all_views)
                    for view_name in VIEW_NAMES:
                        views_out[view_name].append(views[view_name][0].numpy())
                    if name == next(iter(usable)):
                        ground_truth.append(load_ground_truth(args.annotations_root, scenario,
                                                              case["frame_id"]))
                        scenarios.append(scenario)
                    done += 1
                    if done % 20 == 0 or done == len(sample):
                        print(f"        {done}/{len(sample)} casos "
                              f"({time.time() - t_start:.0f}s transcurridos)", flush=True)
        elapsed = time.time() - t_start
        per_checkpoint_views[name] = views_out
        per_checkpoint_time[name] = elapsed
        print(f"        {elapsed:.1f}s total, {elapsed / len(sample):.3f}s/caso "
              f"({len(VIEW_NAMES)} vistas/caso, temperature={temperature:.3f})")

    ground_truth_arr = np.stack(ground_truth)
    scenarios_arr = np.asarray(scenarios)

    print("[5/5] Componiendo configuraciones desde las vistas ya calculadas y evaluando...")

    def mean_logits(checkpoint_names: list[str], view_names: list[str]) -> np.ndarray:
        stacks = []
        for cp in checkpoint_names:
            for vn in view_names:
                stacks.append(np.stack(per_checkpoint_views[cp][vn]))
        return np.mean(stacks, axis=0)

    checkpoint_names = list(usable)
    single_cp = checkpoint_names[:1]
    temperature = float(load_model(usable[single_cp[0]]).heatmap_temperature)

    configs = {
        "baseline_no_tta": (single_cp, ["identity"]),
        "tta_flip_rot_6way": (single_cp, ["identity", "hflip", "vflip", "rot90", "rot180", "rot270"]),
        "tta_scale_3way": (single_cp, ["identity", "zoom_0.90", "zoom_1.10"]),
        "tta_combined_8way": (single_cp, VIEW_NAMES),
    }
    if len(checkpoint_names) > 1:
        configs["ensemble_no_tta"] = (checkpoint_names, ["identity"])
        configs["ensemble_tta_combined"] = (checkpoint_names, VIEW_NAMES)

    results = {
        "n_cases": len(sample),
        "n_per_scenario": args.n_per_scenario,
        "val_scenarios": val_scenarios,
        "checkpoint_candidates_triage": triage,
        "checkpoints_used": checkpoint_names,
        "per_checkpoint_seconds_total": per_checkpoint_time,
        "per_checkpoint_seconds_per_case_all_views": {
            k: v / len(sample) for k, v in per_checkpoint_time.items()
        },
        "seconds_per_view_per_case": {
            k: v / len(sample) / len(VIEW_NAMES) for k, v in per_checkpoint_time.items()
        },
        "configs": {},
    }

    for config_name, (cps, view_subset) in configs.items():
        fused = mean_logits(cps, view_subset)
        fused_t = torch.from_numpy(fused)
        preds = []
        for i in range(fused_t.shape[0]):
            preds.append(decode_xy(fused_t[i:i + 1], temperature))
        preds_arr = np.stack(preds)
        metrics = evaluate_keypoints(preds_arr, ground_truth_arr, scenarios_arr)
        n_views = len(view_subset)
        n_cps = len(cps)
        per_case_seconds = sum(
            per_checkpoint_time[cp] / len(sample) * (n_views / len(VIEW_NAMES)) for cp in cps
        )
        results["configs"][config_name] = {
            "checkpoints": cps,
            "views": view_subset,
            "n_forward_passes_per_case": n_views * n_cps,
            "auc": metrics["auc"],
            "mean_error": metrics["mean_error"],
            "n": metrics["n"],
            "per_scenario": {s: {"n": v["n"], "auc": v["auc"], "mean_error": v["mean_error"]}
                             for s, v in metrics["per_scenario"].items()},
            "measured_seconds_per_case": per_case_seconds,
        }
        print(f"      {config_name}: auc={metrics['auc']:.4f} "
              f"mean_error={metrics['mean_error']:.2f}px "
              f"({n_views * n_cps} forward/caso, ~{per_case_seconds:.3f}s/caso medidos en CPU)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nResultados escritos en {args.output}")


if __name__ == "__main__":
    main()
