"""¿La posición del instrumento en la imagen del microscopio (fundus, 2D,
mismo dominio que ve el modelo) predice el CENTRO del recorte OCT de Task 2?

Distinto del oráculo cross-modal ya descartado (T2-R11/T2-R11b, en
`measure_task2_oracle_signal.py`): aquél proyectaba los keypoints del
instrumento AL VOLUMEN OCT vía la matriz GT y comparaba con la segmentación
del instrumento DENTRO del volumen -- una pregunta de correspondencia
cross-modal. Esta pregunta es puramente 2D: ¿la punta del instrumento tal
como aparece en `Stereo Left/*/microscope.png` (`Keypoints/Forceps`,
`Keypoints/Endoilluminator`) está cerca del centro del crosshair anotado
(`matrix[:2,2]`), EN LA MISMA IMAGEN, sin pasar por el volumen?

Surgió al inspeccionar visualmente los renders de R2 (ver
`experiments/97-t2-crosshair-recheck/renders/*_overlay.png`): en varios casos
la punta del fórceps cae a ~20 px del centro del crosshair.

Reporta:
  1. Distancia pareada (candidato -> centro GT) vs distancia nula (shuffle:
     candidato del caso i contra centro GT de un caso j != i), con test de
     permutación sobre la media.
  2. AUC de esquina (idéntica a la oficial) de tres esquemas:
       - "techo posición": centro = GT, rotación/escala = media del dataset.
       - "instrumento": centro = candidato del instrumento, rotación/escala
         = media del dataset.
       - "nulo": centro = shuffle (candidato de otro caso), rotación/escala
         = media del dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fido.geometry import compose_similarity, corner_auc, corner_error, decompose_similarity  # noqa: E402

RNG_SEED = 20260819
N_SHUFFLE = 2000


def find_cases(root: Path) -> list[Path]:
    numerical_dir = root / "Numerical"
    if not numerical_dir.is_dir():
        return []
    return sorted(numerical_dir.glob("*.json"))


def load_all(dataset_roots: list[Path]) -> list[dict]:
    rows = []
    for root in dataset_roots:
        for json_path in find_cases(root):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
            kp = data.get("Keypoints", {})
            forceps = kp.get("Forceps", {})
            endo = kp.get("Endoilluminator", {})
            rows.append({
                "case_id": f"{root.name}/{json_path.stem}",
                "matrix": matrix,
                "forceps": forceps,
                "endo": endo,
            })
    return rows


def candidate_points(row: dict) -> dict[str, np.ndarray]:
    out = {}
    f = row["forceps"]
    if f.get("Right Head Tip") is not None and f.get("Left Head Tip") is not None:
        out["forceps_midtip"] = (np.array(f["Right Head Tip"]) + np.array(f["Left Head Tip"])) / 2.0
    if f.get("Joint Tip") is not None:
        out["forceps_joint"] = np.array(f["Joint Tip"])
    pts = [np.array(f[k]) for k in ("Right Head Tip", "Left Head Tip", "Joint Tip") if f.get(k) is not None]
    if pts:
        out["forceps_centroid"] = np.mean(pts, axis=0)
    e = row["endo"]
    if e.get("Tip") is not None:
        out["endo_tip"] = np.array(e["Tip"])
    return out


def best_candidate(row: dict, priority: list[str]) -> tuple[str, np.ndarray] | None:
    cands = candidate_points(row)
    for name in priority:
        if name in cands:
            return name, cands[name]
    return None


def paired_vs_shuffle(rows: list[dict], key_fn, rng: np.random.Generator) -> dict:
    centers = np.array([r["matrix"][:2, 2] for r in rows])
    picks = []
    for r in rows:
        picked = key_fn(r)
        picks.append(picked[1] if picked is not None else None)
    valid = [i for i, p in enumerate(picks) if p is not None]
    if len(valid) < 5:
        return {"n": len(valid)}
    cand = np.array([picks[i] for i in valid])
    truth = centers[valid]
    d_true = np.linalg.norm(cand - truth, axis=1)

    shuffle_means = []
    idx = np.arange(len(valid))
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(len(valid))
        # evita empates i==perm[i]
        while np.any(perm == idx):
            perm = rng.permutation(len(valid))
        d_null = np.linalg.norm(cand - truth[perm], axis=1)
        shuffle_means.append(d_null.mean())
    shuffle_means = np.array(shuffle_means)
    p_value = float(np.mean(shuffle_means <= d_true.mean()))  # una cola: ¿es la real tan chica o mas?

    return {
        "n": len(valid),
        "mean_true_dist_px": float(d_true.mean()),
        "median_true_dist_px": float(np.median(d_true)),
        "std_true_dist_px": float(d_true.std()),
        "q25_px": float(np.percentile(d_true, 25)),
        "q75_px": float(np.percentile(d_true, 75)),
        "shuffle_null_mean_of_means_px": float(shuffle_means.mean()),
        "shuffle_null_std_of_means_px": float(shuffle_means.std()),
        "p_value_perm (P[null_mean <= true_mean])": p_value,
        "frac_within_50px": float(np.mean(d_true <= 50)),
        "frac_within_100px": float(np.mean(d_true <= 100)),
        "coverage_frac_cases_with_candidate": len(valid) / len(rows),
    }


def auc_scheme(rows: list[dict], translation_fn, mean_cos: float, mean_sin: float,
                mean_scale: float) -> dict:
    errors = []
    n_missing = 0
    for r in rows:
        t = translation_fn(r)
        if t is None:
            n_missing += 1
            continue
        pred = compose_similarity(t[0], t[1], mean_cos, mean_sin, mean_scale, reflect=True)
        err = corner_error(pred, r["matrix"])
        errors.append(float(err))
    errors = np.asarray(errors)
    return {
        "n": len(errors), "n_missing_candidate": n_missing,
        "mean_error_px": float(errors.mean()) if len(errors) else np.nan,
        "median_error_px": float(np.median(errors)) if len(errors) else np.nan,
        "corner_auc": float(corner_auc(errors)) if len(errors) else np.nan,
    }


def main() -> None:
    train_roots = [REPO_ROOT / "data" / "Task 2" / "Scenario_03"]
    cache = REPO_ROOT / "experiments" / "97-t2-crosshair-recheck" / "cache"
    train_roots += sorted(cache.glob("Scenario_*"))
    mock_root = REPO_ROOT / "data" / "Mock Test" / "Task 2" / "Scenario_12"

    train_rows = load_all(train_roots)
    mock_rows = load_all([mock_root])
    print(f"train: {len(train_rows)} casos | mock: {len(mock_rows)} casos")

    # cobertura de cada candidato
    for tag, rows in (("train", train_rows), ("mock", mock_rows)):
        cov = {}
        for r in rows:
            for name in candidate_points(r):
                cov[name] = cov.get(name, 0) + 1
        print(f"[{tag}] cobertura candidatos (de {len(rows)}): {cov}")

    priority = ["forceps_midtip", "forceps_joint", "forceps_centroid", "endo_tip"]

    rng = np.random.default_rng(RNG_SEED)
    print("\n=== A. Distancia pareada vs nula (shuffle), set TRAIN ===")
    for name in ["forceps_midtip", "forceps_joint", "forceps_centroid", "endo_tip"]:
        res = paired_vs_shuffle(train_rows, lambda r, n=name: (n, candidate_points(r)[n]) if n in candidate_points(r) else None, rng)
        print(f"  {name}: {json.dumps(res, indent=2)}")

    print("\n  combinado (prioridad forceps_midtip > forceps_joint > forceps_centroid > endo_tip):")
    res_combo = paired_vs_shuffle(train_rows, lambda r: best_candidate(r, priority), rng)
    print(f"  {json.dumps(res_combo, indent=2)}")

    print("\n=== A (mock). Distancia pareada vs nula, set MOCK ===")
    res_combo_mock = paired_vs_shuffle(mock_rows, lambda r: best_candidate(r, priority), np.random.default_rng(RNG_SEED + 1))
    print(f"  combinado: {json.dumps(res_combo_mock, indent=2)}")

    # estadisticos globales de rotacion/escala (para el "techo" con rot/escala media)
    decomp = [decompose_similarity(r["matrix"], reflect=True) for r in train_rows]
    cos_vals = np.array([d["cos_theta"] for d in decomp])
    sin_vals = np.array([d["sin_theta"] for d in decomp])
    scale_vals = np.array([d["scale"] for d in decomp])
    # normalizar (cos,sin) para que corresponda a un angulo medio valido
    ang = np.arctan2(sin_vals, cos_vals)
    mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    mean_cos, mean_sin = float(np.cos(mean_ang)), float(np.sin(mean_ang))
    mean_scale = float(scale_vals.mean())
    print(f"\nEscala: media={scale_vals.mean():.2f} std={scale_vals.std():.2f} "
          f"({100*scale_vals.std()/scale_vals.mean():.1f}% rel)")
    print(f"Angulo: circ_std~={np.degrees(np.sqrt(-2*np.log(np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))))):.1f} deg")

    print("\n=== B. AUC de esquina por esquema, set TRAIN ===")
    schemes = {
        "techo_posicion_GT": lambda r: tuple(r["matrix"][:2, 2]),
        "instrumento_combo": lambda r: (lambda bc: tuple(bc[1]) if bc else None)(best_candidate(r, priority)),
        "forceps_midtip_solo": lambda r: (lambda c: tuple(c["forceps_midtip"]) if "forceps_midtip" in c else None)(candidate_points(r)),
    }
    for name, fn in schemes.items():
        res = auc_scheme(train_rows, fn, mean_cos, mean_sin, mean_scale)
        print(f"  {name}: {json.dumps(res, indent=2)}")

    # nulo: centro = candidato de OTRO caso (shuffle fijo con la misma seed)
    rng2 = np.random.default_rng(RNG_SEED + 7)
    perm = rng2.permutation(len(train_rows))
    while np.any(perm == np.arange(len(train_rows))):
        perm = rng2.permutation(len(train_rows))

    def null_fn(r, _perm=perm, _rows=train_rows, _idx={"i": 0}):
        return None  # se resuelve abajo con índice explícito

    errors_null = []
    for i, r in enumerate(train_rows):
        bc = best_candidate(train_rows[perm[i]], priority)
        if bc is None:
            continue
        pred = compose_similarity(bc[1][0], bc[1][1], mean_cos, mean_sin, mean_scale, reflect=True)
        errors_null.append(float(corner_error(pred, r["matrix"])))
    errors_null = np.asarray(errors_null)
    print(f"  nulo_instrumento_shuffle: n={len(errors_null)} "
          f"mean_error_px={errors_null.mean():.2f} corner_auc={corner_auc(errors_null):.4f}")

    print("\n=== B (mock). AUC de esquina, set MOCK (mismos mean_cos/sin/scale de TRAIN) ===")
    for name, fn in schemes.items():
        res = auc_scheme(mock_rows, fn, mean_cos, mean_sin, mean_scale)
        print(f"  {name}: {json.dumps(res, indent=2)}")


if __name__ == "__main__":
    main()
