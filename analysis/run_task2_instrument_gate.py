"""T2-89 gate: Procrustes/Umeyama GT-a-GT sobre los casos con >=2 keypoints de
instrumento dentro de la huella en-face.

Mide el LIMITE SUPERIOR de la via de instrumento (usa la matriz GT para
proyectar los keypoints de fundus a uv y para filtrar "dentro de la huella";
el punto del lado OCT es el pixel segmentado (GT, clase instrumento) mas
cercano a esa proyeccion GT). NO mide lo alcanzable en inferencia: un solver
real necesitaria ademas detectar el instrumento en ambas modalidades sin usar
la matriz GT, y esos detectores no existen todavia (ver INFORME.md, S4.2).

Todo local: las anotaciones vienen de data/_annotations/Task 2 (JSON) y la
segmentacion del volumen se lee directamente de los zips de
data/Task 2/Scenario_NN.zip via zipfile, sin extraer el archivo completo.
No requiere GPU ni el pod de RunPod.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from analysis.verify_task2_enface_convention import (  # noqa: E402
    INSTRUMENT_CLASSES, map_fundus_to_uv, points_from_group, transform_uv,
)
from fido.geometry import corner_auc, corner_error  # noqa: E402

CONVENTION = dict(transpose=True, flip_u=True, flip_v=True)  # transpose__flip_u__flip_v, congelada


def fit_reflected_similarity(ref_uv: np.ndarray, pred_pixels: np.ndarray) -> np.ndarray:
    """Ajuste lineal por minimos cuadrados de la similitud reflejada de 4 DOF
    (misma familia que `compose_similarity(..., reflect=True)` en
    `fido/geometry.py`: A = [[a, b], [b, -a]], det(A) = -(a^2+b^2) < 0).

    NO se usa `fido.geometry.fit_closed_form_similarity` aqui: esa funcion
    ajusta una ROTACION PROPIA por SVD (Kabsch clasico, u@vt sin restringir
    signo) y depende de que la reconstruccion posterior con
    `compose_similarity(reflect=True)` reinterprete el angulo recuperado como
    si fuera el de la similitud reflejada -- una reinterpretacion, no un
    ajuste directo. Con exactamente 2 puntos la covarianza es de rango 1 y el
    signo de la rotacion queda indeterminado por los datos (rotacion propia e
    impropia ajustan igual de bien los 2 puntos, pero divergen en cualquier
    otro punto). Verificado con un caso sintetico exacto (M conocida,
    reflejada, 2 puntos): `fit_closed_form_similarity` + recomponer con
    `reflect=True` da 138 px de error residual donde debería dar 0.

    Esta funcion en cambio impone la estructura fija A=[[a,b],[b,-a]] como
    restriccion LINEAL del ajuste (a,b,tx,ty entran linealmente en las
    ecuaciones), lo que evita la ambiguedad de reflexion por construccion.
    Verificado exacto (error ~1e-12 px) en 2 y 3 puntos sinteticos sin ruido.
    """
    k = len(ref_uv)
    design = np.zeros((2 * k, 4))
    target = np.zeros(2 * k)
    for i, (u, v) in enumerate(ref_uv):
        design[2 * i] = [u, v, 1.0, 0.0]
        target[2 * i] = pred_pixels[i, 0]
        design[2 * i + 1] = [-v, u, 0.0, 1.0]
        target[2 * i + 1] = pred_pixels[i, 1]
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    a, b, tx, ty = params
    return np.array([[a, b, tx], [b, -a, ty], [0.0, 0.0, 1.0]])


def load_instrument_mask_from_zip(zf, frame_id):
    """Replica load_volume_label_maps + seg==INSTRUMENT_CLASSES, axis=1, leyendo
    directamente del zip (sin extraer). Devuelve None si el frame no tiene
    carpeta Segmentation en el zip."""
    prefix = "iOCT Microscope/Volume/" + frame_id + "/Segmentation/"
    names = sorted(n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png"))
    if not names:
        return None
    slices = []
    for name in names:
        data = zf.read(name)
        slices.append(np.array(Image.open(io.BytesIO(data))))
    seg = np.stack(slices, axis=0)  # (n_slices, depth, width)
    return np.any(np.isin(seg, INSTRUMENT_CLASSES), axis=1)  # (n_slices, width)


def nearest_instrument_point(matrix_uv_point, indices, h, w):
    """matrix_uv_point: (u,v) en el dominio de la matriz GT. Devuelve el punto
    (u,v), mismo dominio, del pixel de instrumento segmentado mas cercano en
    el volumen -- via la orientacion nativa (convencion congelada) y de vuelta.
    """
    native = transform_uv(matrix_uv_point.reshape(1, 2), **CONVENTION)[0]
    x = int(np.clip(round(native[0] * (w - 1)), 0, w - 1))
    y = int(np.clip(round(native[1] * (h - 1)), 0, h - 1))
    ny, nx = indices[0, y, x], indices[1, y, x]
    native_near = np.array([nx / (w - 1), ny / (h - 1)])
    # La convencion transpose__flip_u__flip_v es autoinversa (C = C^-1, ver
    # PRE_REGISTRATION.md de 83-t2-enface-convention) -> misma funcion vuelve.
    matrix_near = transform_uv(native_near.reshape(1, 2), **CONVENTION)[0]
    return matrix_near


def instrument_points_for_case(data):
    forceps_names = ("Right Head Tip", "Left Head Tip", "Joint Tip", "Start")
    illum_names = ("Tip", "Start")
    forceps = points_from_group(data, "Endgripping Forceps", forceps_names)
    forceps_labels = ["Forceps:" + k for k in forceps_names
                      if data.get("Keypoints", {}).get("Endgripping Forceps", {}).get(k) is not None]
    illum = points_from_group(data, "Endoilluminator", illum_names)
    illum_labels = ["Illuminator:" + k for k in illum_names
                   if data.get("Keypoints", {}).get("Endoilluminator", {}).get(k) is not None]
    points = np.concatenate([forceps, illum]) if (len(forceps) or len(illum)) else np.empty((0, 2))
    labels = forceps_labels + illum_labels
    return points, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--zips-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    all_case_ids = []
    candidates = []  # cases with >=2 points inside footprint

    for scenario_dir in sorted(args.annotations_root.glob("Scenario_*")):
        for json_path in sorted(scenario_dir.glob("*.json")):
            case_id = scenario_dir.name + "/" + json_path.stem
            all_case_ids.append(case_id)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
            fundus_pts, labels = instrument_points_for_case(data)
            if len(fundus_pts) == 0:
                continue
            matrix_uv = map_fundus_to_uv(fundus_pts, matrix)
            valid = np.all((matrix_uv >= 0.0) & (matrix_uv <= 1.0), axis=1)
            if valid.sum() < 2:
                continue
            candidates.append({
                "case_id": case_id, "scenario": scenario_dir.name, "frame_id": json_path.stem,
                "matrix": matrix, "fundus_pts": fundus_pts[valid], "matrix_uv": matrix_uv[valid],
                "labels": [l for l, v in zip(labels, valid) if v],
            })

    print("casos totales: " + str(len(all_case_ids)))
    print("candidatos (>=2 puntos dentro de la huella): " + str(len(candidates)))

    results = []
    zip_cache = {}
    for candidate in candidates:
        scenario = candidate["scenario"]
        if scenario not in zip_cache:
            zip_path = args.zips_root / (scenario + ".zip")
            if not zip_path.exists():
                print("FALTA " + str(zip_path) + " -- no se puede continuar con este escenario", flush=True)
                zip_cache[scenario] = None
            else:
                zip_cache[scenario] = zipfile.ZipFile(zip_path)
        zf = zip_cache[scenario]
        record = {"case_id": candidate["case_id"], "scenario": scenario,
                  "n_points": len(candidate["fundus_pts"]), "status": None, "corner_error_px": None}
        if zf is None:
            record["status"] = "zip_missing"
            results.append(record)
            continue
        mask = load_instrument_mask_from_zip(zf, candidate["frame_id"])
        if mask is None:
            record["status"] = "no_segmentation_in_zip"
            results.append(record)
            continue
        if not mask.any():
            record["status"] = "empty_instrument_mask"
            results.append(record)
            continue
        h, w = mask.shape
        dist, indices = distance_transform_edt(
            ~mask, sampling=(1.0 / (h - 1), 1.0 / (w - 1)), return_indices=True)
        oct_pts = np.stack([
            nearest_instrument_point(uv, indices, h, w) for uv in candidate["matrix_uv"]
        ])
        m_hat = fit_reflected_similarity(ref_uv=oct_pts, pred_pixels=candidate["fundus_pts"])
        err = float(corner_error(m_hat, candidate["matrix"]))
        record["status"] = "ok"
        record["corner_error_px"] = err
        record["labels"] = candidate["labels"]
        results.append(record)
        print("[" + str(len(results)) + "/" + str(len(candidates)) + "] " + candidate["case_id"] +
             " n=" + str(record["n_points"]) + " err=" + ("%.2f" % err) + "px", flush=True)

    for zf in zip_cache.values():
        if zf is not None:
            zf.close()

    ok_errors = np.array([r["corner_error_px"] for r in results if r["status"] == "ok"])
    n_ok = len(ok_errors)
    n_candidates = len(candidates)
    print("\ncandidatos OK (score calculado): " + str(n_ok) + "/" + str(n_candidates))
    status_counts = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    print("desglose de status: " + str(status_counts))

    if n_ok:
        auc_subset = corner_auc(ok_errors)
        mean_subset = float(np.mean(ok_errors))
        median_subset = float(np.median(ok_errors))
    else:
        auc_subset = mean_subset = median_subset = float("nan")
    print("\nAUC sobre el subconjunto OK (n=" + str(n_ok) + "): " + ("%.4f" % auc_subset))
    print("error medio px (subconjunto OK): " + ("%.2f" % mean_subset))
    print("error mediano px (subconjunto OK): " + ("%.2f" % median_subset))

    # Extrapolacion a los 1214: casos sin score (no candidato, o candidato
    # fallido) puntuan 0 en TODOS los umbrales (equivalente a error=+inf).
    ok_case_ids = {r["case_id"] for r in results if r["status"] == "ok"}
    ok_error_by_id = {r["case_id"]: r["corner_error_px"] for r in results if r["status"] == "ok"}
    full_errors = np.array([
        ok_error_by_id[cid] if cid in ok_case_ids else np.inf
        for cid in all_case_ids
    ])
    auc_full = corner_auc(full_errors)
    print("\nAUC extrapolado a los " + str(len(all_case_ids)) +
         " casos (0 en los que no llegan a n_ok): " + ("%.4f" % auc_full))

    print("\ndesglose por escenario (candidatos / OK / AUC-escenario):")
    per_scenario = {}
    for cid in all_case_ids:
        scen = cid.split("/")[0]
        per_scenario.setdefault(scen, {"total": 0, "candidates": 0, "ok": 0, "errors": []})
        per_scenario[scen]["total"] += 1
    for candidate in candidates:
        per_scenario[candidate["scenario"]]["candidates"] += 1
    for r in results:
        if r["status"] == "ok":
            per_scenario[r["scenario"]]["ok"] += 1
            per_scenario[r["scenario"]]["errors"].append(r["corner_error_px"])
    for scen in sorted(per_scenario):
        s = per_scenario[scen]
        errs = np.array(s["errors"])
        auc_s = corner_auc(errs) if len(errs) else float("nan")
        mean_s = float(np.mean(errs)) if len(errs) else float("nan")
        print("  " + scen + ": total=" + str(s["total"]) + " candidatos=" + str(s["candidates"]) +
             " ok=" + str(s["ok"]) + " AUC=" + ("%.4f" % auc_s) + " error_medio=" + ("%.2f" % mean_s) + "px")

    payload = {
        "n_total_cases": len(all_case_ids), "n_candidates": n_candidates, "n_ok": n_ok,
        "status_counts": status_counts,
        "auc_subset_ok": None if np.isnan(auc_subset) else auc_subset,
        "mean_error_subset_px": None if np.isnan(mean_subset) else mean_subset,
        "median_error_subset_px": None if np.isnan(median_subset) else median_subset,
        "auc_extrapolated_1214": auc_full,
        "per_scenario": {
            scen: {"total": s["total"], "candidates": s["candidates"], "ok": s["ok"],
                  "auc": (corner_auc(np.array(s["errors"])) if s["errors"] else None),
                  "mean_error_px": (float(np.mean(s["errors"])) if s["errors"] else None)}
            for scen, s in per_scenario.items()
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
