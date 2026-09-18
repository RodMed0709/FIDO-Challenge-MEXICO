"""R2 revisitado con números: ¿es detectable el crosshair/rectángulo de
escaneo del iOCT en la imagen del microscopio (`opmi_image` / `microscope.png`)?

El cierre previo de R2 ("LOGRADO -- NO") no tenía ni un número. Este script
corre cuatro pruebas cuantitativas sobre la posición anotada por el simulador
(`Keypoints/iOCT Microscope Crosshair` y la matriz `Ground Truth/Task 2`,
vía `project_corners`) contra controles nulos en la MISMA imagen:

  A. Fotometría en las líneas del crosshair vs píxeles de control aleatorios
     (por canal RGB, contraste local, magnitud de borde Sobel).
  B. Detección de líneas rectas (Canny + Hough) cerca de la posición anotada
     vs posiciones nulas.
  C. Estadísticos de la región escaneada (proyección del cuadrado unitario
     vía `project_corners`) vs regiones nulas de la misma forma/tamaño
     trasladadas aleatoriamente dentro de la imagen.
  D. Renders para inspección visual (guardados aparte, no aquí).

Reporta SIEMPRE el delta pareado (caso por caso, GT - control) con Wilcoxon
signed-rank + Cohen's d, nunca solo el promedio. Se corre igual sobre el set
"train" (Scenario_01/03/05/08/10) y sobre el Mock Test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fido.geometry import project_corners  # noqa: E402

RNG_SEED = 20260819
N_NULL_HOUGH = 20
N_NULL_REGION = 30
IMG_SIZE = 1024
MARGIN_FROM_CROSSHAIR_PX = 25  # separación mínima control-fotometría / líneas
IMG_CENTER = np.array([(IMG_SIZE - 1) / 2.0, (IMG_SIZE - 1) / 2.0])
RADIAL_JITTER_PX = 15.0  # tolerancia de radio para el control "matched"


def find_cases(root: Path) -> list[dict]:
    """Empareja Numerical/<fid>.json con Stereo Left/<fid>/microscope.png
    bajo `root` (una carpeta de escenario ya con ambas subcarpetas)."""
    cases = []
    numerical_dir = root / "Numerical"
    if not numerical_dir.is_dir():
        return cases
    for json_path in sorted(numerical_dir.glob("*.json")):
        fid = json_path.stem
        img_path = root / "Stereo Left" / fid / "microscope.png"
        if img_path.is_file():
            cases.append({"scenario": root.name, "frame_id": fid,
                          "json_path": json_path, "img_path": img_path})
    return cases


def load_case(case: dict) -> tuple[dict, np.ndarray]:
    data = json.loads(case["json_path"].read_text(encoding="utf-8"))
    img_bgr = cv2.imread(str(case["img_path"]), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"no se pudo leer {case['img_path']}")
    return data, img_bgr


def crosshair_segments(data: dict) -> list[np.ndarray]:
    ch = data.get("Keypoints", {}).get("iOCT Microscope Crosshair", {})
    segs = []
    for a, b in (("Start 0", "End 0"), ("Start 1", "End 1")):
        pa, pb = ch.get(a), ch.get(b)
        if pa is not None and pb is not None:
            segs.append(np.array([pa, pb], dtype=np.float64))
    return segs


def bilinear_sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """img: HxWxC uint8/float. xy: (N,2) en (x,y). Devuelve (N,C) float64."""
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0
    img_f = img.astype(np.float64)
    c00, c10 = img_f[y0, x0], img_f[y0, x1]
    c01, c11 = img_f[y1, x0], img_f[y1, x1]
    top = c00 * (1 - wx[:, None]) + c10 * wx[:, None]
    bot = c01 * (1 - wx[:, None]) + c11 * wx[:, None]
    return top * (1 - wy[:, None]) + bot * wy[:, None]


def points_on_segments(segs: list[np.ndarray], n_per_seg: int,
                        skip_center_frac: float = 0.08) -> np.ndarray:
    """Puntos a lo largo de cada segmento, saltando `skip_center_frac`
    alrededor de t=0.5 (donde ambos brazos se cruzan) para no contar el
    centro dos veces con peso distinto."""
    pts = []
    ts = np.linspace(0.03, 0.97, n_per_seg)
    ts = ts[np.abs(ts - 0.5) > skip_center_frac]
    for seg in segs:
        p0, p1 = seg
        for t in ts:
            pts.append(p0 * (1 - t) + p1 * t)
    return np.asarray(pts, dtype=np.float64)


def random_control_points(n: int, segs: list[np.ndarray], rng: np.random.Generator,
                           size: int = IMG_SIZE, margin: float = MARGIN_FROM_CROSSHAIR_PX,
                           border: float = 15.0) -> np.ndarray:
    """`n` puntos uniformes en la imagen, lejos de las líneas del crosshair y
    del borde (para que la interpolación bilineal sea válida)."""
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        p = rng.uniform(border, size - 1 - border, size=2)
        ok = True
        for seg in segs:
            d = point_segment_distance(p, seg[0], seg[1])
            if d < margin:
                ok = False
                break
        if ok:
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def radius_matched_control_points(ch_pts: np.ndarray, segs: list[np.ndarray],
                                   rng: np.random.Generator, size: int = IMG_SIZE,
                                   margin: float = MARGIN_FROM_CROSSHAIR_PX,
                                   border: float = 15.0,
                                   radial_jitter: float = RADIAL_JITTER_PX) -> np.ndarray:
    """Un punto de control por cada punto del crosshair, a la MISMA distancia
    del centro de la imagen (±`radial_jitter`) pero ángulo aleatorio.

    Las imágenes de fundus tienen viñeteado fuerte (el centro óptico es
    brillante, los bordes/esquinas son negros); un control uniforme sobre
    toda la imagen confunde "cerca del centro óptico" con "cerca del
    crosshair". Este control aísla la señal del crosshair de la excentricidad
    radial."""
    out = np.empty_like(ch_pts)
    for i, p in enumerate(ch_pts):
        r = float(np.linalg.norm(p - IMG_CENTER))
        placed = False
        for _ in range(300):
            r_try = r + rng.uniform(-radial_jitter, radial_jitter)
            theta = rng.uniform(0, 2 * np.pi)
            cand = IMG_CENTER + r_try * np.array([np.cos(theta), np.sin(theta)])
            if not (border <= cand[0] <= size - 1 - border and border <= cand[1] <= size - 1 - border):
                continue
            if any(point_segment_distance(cand, seg[0], seg[1]) < margin for seg in segs):
                continue
            out[i] = cand
            placed = True
            break
        if not placed:
            out[i] = p  # no debería pasar casi nunca; se filtra al ser igual al propio punto
    return out


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def local_contrast(gray: np.ndarray, xy: np.ndarray, half: int = 3) -> np.ndarray:
    h, w = gray.shape
    out = np.empty(len(xy))
    for i, (x, y) in enumerate(xy):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - half), min(w, xi + half + 1)
        y0, y1 = max(0, yi - half), min(h, yi + half + 1)
        patch = gray[y0:y1, x0:x1]
        out[i] = patch.std() if patch.size else np.nan
    return out


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


# ---------------------------------------------------------------------------
# A. Fotometría crosshair vs control
# ---------------------------------------------------------------------------

def photometry_case(data: dict, img_bgr: np.ndarray, rng: np.random.Generator,
                     n_per_seg: int = 25) -> dict | None:
    segs = crosshair_segments(data)
    if len(segs) != 2:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    edges = sobel_magnitude(gray)

    ch_pts = points_on_segments(segs, n_per_seg)
    ctrl_pts = random_control_points(len(ch_pts), segs, rng)
    matched_pts = radius_matched_control_points(ch_pts, segs, rng)
    if len(ctrl_pts) < len(ch_pts) * 0.5:
        return None

    ch_rgb = bilinear_sample(img_bgr, ch_pts)[:, ::-1]   # BGR->RGB
    ctrl_rgb = bilinear_sample(img_bgr, ctrl_pts)[:, ::-1]
    matched_rgb = bilinear_sample(img_bgr, matched_pts)[:, ::-1]
    ch_contrast = local_contrast(gray, ch_pts)
    ctrl_contrast = local_contrast(gray, ctrl_pts)
    matched_contrast = local_contrast(gray, matched_pts)
    ch_edge = bilinear_sample(edges[:, :, None], ch_pts)[:, 0]
    ctrl_edge = bilinear_sample(edges[:, :, None], ctrl_pts)[:, 0]
    matched_edge = bilinear_sample(edges[:, :, None], matched_pts)[:, 0]

    out = {
        "delta_R": float(ch_rgb[:, 0].mean() - ctrl_rgb[:, 0].mean()),
        "delta_G": float(ch_rgb[:, 1].mean() - ctrl_rgb[:, 1].mean()),
        "delta_B": float(ch_rgb[:, 2].mean() - ctrl_rgb[:, 2].mean()),
        "delta_contrast": float(np.nanmean(ch_contrast) - np.nanmean(ctrl_contrast)),
        "delta_edge": float(np.nanmean(ch_edge) - np.nanmean(ctrl_edge)),
        "matched_delta_R": float(ch_rgb[:, 0].mean() - matched_rgb[:, 0].mean()),
        "matched_delta_G": float(ch_rgb[:, 1].mean() - matched_rgb[:, 1].mean()),
        "matched_delta_B": float(ch_rgb[:, 2].mean() - matched_rgb[:, 2].mean()),
        "matched_delta_contrast": float(np.nanmean(ch_contrast) - np.nanmean(matched_contrast)),
        "matched_delta_edge": float(np.nanmean(ch_edge) - np.nanmean(matched_edge)),
    }
    return out


# ---------------------------------------------------------------------------
# B. Hough
# ---------------------------------------------------------------------------

def segment_match(cand: np.ndarray, target: np.ndarray, dist_tol: float = 8.0,
                   angle_tol_deg: float = 10.0) -> bool:
    d1 = min(point_segment_distance(cand[0], target[0], target[1]),
              point_segment_distance(cand[1], target[0], target[1]))
    d2 = min(point_segment_distance(target[0], cand[0], cand[1]),
              point_segment_distance(target[1], cand[0], cand[1]))
    if max(d1, d2) > dist_tol:
        return False
    v1 = cand[1] - cand[0]
    v2 = target[1] - target[0]
    ang = np.degrees(np.arccos(
        np.clip(abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))
    return ang < angle_tol_deg


def hough_lines(gray: np.ndarray) -> list[np.ndarray]:
    med = np.median(gray)
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med + 30))
    edges = cv2.Canny(gray.astype(np.uint8), lo, hi)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                             minLineLength=25, maxLineGap=4)
    if lines is None:
        return []
    return [np.array([[l[0][0], l[0][1]], [l[0][2], l[0][3]]], dtype=np.float64)
            for l in lines]


def hough_case(data: dict, img_bgr: np.ndarray, rng: np.random.Generator) -> dict | None:
    segs = crosshair_segments(data)
    if len(segs) != 2:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detected = hough_lines(gray)

    gt_hit = any(segment_match(d, seg) for d in detected for seg in segs)

    center = (segs[0][0] + segs[0][1] + segs[1][0] + segs[1][1]) / 4.0
    null_hits = 0
    for _ in range(N_NULL_HOUGH):
        offset = rng.uniform(-400, 400, size=2)
        new_center = center + offset
        new_center = np.clip(new_center, 60, IMG_SIZE - 60)
        shift = new_center - center
        null_segs = [seg + shift for seg in segs]
        null_hits += int(any(segment_match(d, seg) for d in detected for seg in null_segs))

    # Nulo "matched": mismo radio de excentricidad al centro óptico que el
    # centro real del crosshair (las venas/vasos y bordes de instrumento se
    # concentran cerca del centro de la retina visible, no uniformemente;
    # esto evita que ese sesgo por sí solo produzca "hits").
    r_gt = float(np.linalg.norm(center - IMG_CENTER))
    matched_hits = 0
    matched_trials = 0
    for _ in range(N_NULL_HOUGH):
        r_try = r_gt + rng.uniform(-25.0, 25.0)
        theta = rng.uniform(0, 2 * np.pi)
        new_center = IMG_CENTER + r_try * np.array([np.cos(theta), np.sin(theta)])
        new_center = np.clip(new_center, 60, IMG_SIZE - 60)
        shift = new_center - center
        null_segs = [seg + shift for seg in segs]
        matched_trials += 1
        matched_hits += int(any(segment_match(d, seg) for d in detected for seg in null_segs))

    return {"gt_hit": bool(gt_hit), "null_hit_rate": null_hits / N_NULL_HOUGH,
            "matched_hit_rate": matched_hits / matched_trials if matched_trials else np.nan,
            "matched_trials": matched_trials,
            "n_detected": len(detected)}


# ---------------------------------------------------------------------------
# C. Región escaneada vs nula
# ---------------------------------------------------------------------------

def region_stats(img_bgr: np.ndarray, gray: np.ndarray, hsv: np.ndarray,
                  corners: np.ndarray) -> dict | None:
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(corners).astype(np.int32)
    cv2.fillPoly(mask, [poly], 1)
    m = mask.astype(bool)
    if m.sum() < 200:
        return None
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    noise = gray.astype(np.float64) - blur.astype(np.float64)
    return {
        "mean_gray": float(gray[m].mean()),
        "std_gray": float(gray[m].std()),
        "mean_sat": float(hsv[:, :, 1][m].mean()),
        "noise_std": float(noise[m].std()),
        "area": int(m.sum()),
    }


def region_case(data: dict, img_bgr: np.ndarray, rng: np.random.Generator) -> dict | None:
    matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
    corners = project_corners(matrix)
    if not np.all(np.isfinite(corners)):
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    gt = region_stats(img_bgr, gray, hsv, corners)
    if gt is None:
        return None

    center = matrix[:2, 2]
    offsets = corners - center
    low = -offsets.min(axis=0) + 2
    high = (IMG_SIZE - 1) - offsets.max(axis=0) - 2
    if np.any(low > high):
        return None

    null_stats = []
    tries = 0
    while len(null_stats) < N_NULL_REGION and tries < N_NULL_REGION * 6:
        tries += 1
        new_center = rng.uniform(low, high)
        null_corners = corners + (new_center - center)
        st = region_stats(img_bgr, gray, hsv, null_corners)
        if st is not None:
            null_stats.append(st)

    # Control "matched": mismo radio de excentricidad que el centro GT
    # respecto al centro óptico de la imagen (controla el viñeteado), ángulo
    # aleatorio. Se descartan candidatos cuyo rectángulo se saldría del
    # cuadro (mismo criterio que arriba, verificado punto a punto).
    r_gt = float(np.linalg.norm(center - IMG_CENTER))
    matched_stats = []
    tries = 0
    while len(matched_stats) < N_NULL_REGION and tries < N_NULL_REGION * 15:
        tries += 1
        r_try = r_gt + rng.uniform(-25.0, 25.0)
        theta = rng.uniform(0, 2 * np.pi)
        new_center = IMG_CENTER + r_try * np.array([np.cos(theta), np.sin(theta)])
        null_corners = corners + (new_center - center)
        if np.any(null_corners < 2) or np.any(null_corners > IMG_SIZE - 3):
            continue
        st = region_stats(img_bgr, gray, hsv, null_corners)
        if st is not None:
            matched_stats.append(st)

    if len(null_stats) < 5:
        return None

    out = {"gt_" + k: v for k, v in gt.items()}
    for k in ("mean_gray", "std_gray", "mean_sat", "noise_std"):
        null_vals = np.array([s[k] for s in null_stats])
        out[f"null_mean_{k}"] = float(null_vals.mean())
        out[f"delta_{k}"] = float(gt[k] - null_vals.mean())
        if len(matched_stats) >= 5:
            matched_vals = np.array([s[k] for s in matched_stats])
            out[f"matched_null_mean_{k}"] = float(matched_vals.mean())
            out[f"matched_delta_{k}"] = float(gt[k] - matched_vals.mean())
        else:
            out[f"matched_null_mean_{k}"] = np.nan
            out[f"matched_delta_{k}"] = np.nan
    out["n_matched_null"] = len(matched_stats)
    return out


# ---------------------------------------------------------------------------
# Agregación / reporte
# ---------------------------------------------------------------------------

def paired_report(deltas: np.ndarray, label: str) -> dict:
    deltas = deltas[np.isfinite(deltas)]
    n = len(deltas)
    if n < 3:
        return {"label": label, "n": n, "mean": np.nan, "cohens_d": np.nan, "p_wilcoxon": np.nan}
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    d = mean / sd if sd > 1e-9 else np.nan
    try:
        stat, p = stats.wilcoxon(deltas)
    except ValueError:
        p = np.nan
    return {"label": label, "n": n, "mean": mean, "std": sd, "cohens_d": d, "p_wilcoxon": float(p)}


def run(dataset_root_pairs: list[tuple[str, Path]], out_json: Path, tag: str) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    photometry_rows, hough_rows, region_rows = [], [], []
    n_cases = 0
    for scenario_name, root in dataset_root_pairs:
        cases = find_cases(root)
        for case in cases:
            try:
                data, img_bgr = load_case(case)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {case['scenario']}/{case['frame_id']}: {exc}")
                continue
            n_cases += 1
            case_id = f"{scenario_name}/{case['frame_id']}"
            ph = photometry_case(data, img_bgr, rng)
            if ph is not None:
                ph["case_id"] = case_id
                photometry_rows.append(ph)
            ho = hough_case(data, img_bgr, rng)
            if ho is not None:
                ho["case_id"] = case_id
                hough_rows.append(ho)
            re = region_case(data, img_bgr, rng)
            if re is not None:
                re["case_id"] = case_id
                region_rows.append(re)

    print(f"[{tag}] casos cargados: {n_cases}  "
          f"fotometria={len(photometry_rows)}  hough={len(hough_rows)}  region={len(region_rows)}")

    report: dict = {"tag": tag, "n_cases_loaded": n_cases}

    report["photometry"] = {}
    for ch in ("R", "G", "B"):
        vals = np.array([r[f"delta_{ch}"] for r in photometry_rows])
        report["photometry"][ch] = paired_report(vals, f"delta_{ch} (0-255)")
    for metric in ("contrast", "edge"):
        vals = np.array([r[f"delta_{metric}"] for r in photometry_rows])
        report["photometry"][metric] = paired_report(vals, f"delta_{metric}")
    report["photometry_matched"] = {}
    for ch in ("R", "G", "B"):
        vals = np.array([r[f"matched_delta_{ch}"] for r in photometry_rows])
        report["photometry_matched"][ch] = paired_report(vals, f"matched_delta_{ch} (0-255)")
    for metric in ("contrast", "edge"):
        vals = np.array([r[f"matched_delta_{metric}"] for r in photometry_rows])
        report["photometry_matched"][metric] = paired_report(vals, f"matched_delta_{metric}")

    def two_prop_test(gt_rate: float, gt_n: int, null_rate: float, null_n: int):
        try:
            from statsmodels.stats.proportion import proportions_ztest
            count = np.array([gt_rate * gt_n, null_rate * null_n])
            nobs = np.array([gt_n, null_n])
            zstat, pval = proportions_ztest(count, nobs)
            return float(zstat), float(pval)
        except Exception:  # noqa: BLE001
            return np.nan, np.nan

    if hough_rows:
        gt_hits = np.array([r["gt_hit"] for r in hough_rows], dtype=float)
        null_rates = np.array([r["null_hit_rate"] for r in hough_rows])
        matched_rates = np.array([r["matched_hit_rate"] for r in hough_rows])
        gt_rate = float(gt_hits.mean())
        null_rate = float(null_rates.mean())
        matched_rate = float(np.nanmean(matched_rates))
        z_naive, p_naive = two_prop_test(gt_rate, len(hough_rows), null_rate, len(hough_rows) * N_NULL_HOUGH)
        z_matched, p_matched = two_prop_test(gt_rate, len(hough_rows), matched_rate, len(hough_rows) * N_NULL_HOUGH)
        # test pareado caso-a-caso (mas conservador: una observacion por caso)
        try:
            stat_cw, p_cw = stats.wilcoxon(gt_hits - null_rates)
        except ValueError:
            p_cw = np.nan
        try:
            stat_cwm, p_cwm = stats.wilcoxon(gt_hits - matched_rates)
        except ValueError:
            p_cwm = np.nan
        report["hough"] = {
            "n": len(hough_rows), "gt_hit_rate": gt_rate,
            "null_hit_rate_uniform": null_rate, "z_uniform": z_naive, "p_uniform": p_naive,
            "null_hit_rate_matched": matched_rate, "z_matched": z_matched, "p_matched": p_matched,
            "p_wilcoxon_case_level_uniform": float(p_cw),
            "p_wilcoxon_case_level_matched": float(p_cwm),
            "mean_n_detected_lines": float(np.mean([r["n_detected"] for r in hough_rows])),
        }
    else:
        report["hough"] = {"n": 0}

    report["region"] = {}
    for metric in ("mean_gray", "std_gray", "mean_sat", "noise_std"):
        vals = np.array([r[f"delta_{metric}"] for r in region_rows])
        report["region"][metric] = paired_report(vals, f"delta_{metric}")
    report["region_matched"] = {}
    for metric in ("mean_gray", "std_gray", "mean_sat", "noise_std"):
        vals = np.array([r[f"matched_delta_{metric}"] for r in region_rows])
        report["region_matched"][metric] = paired_report(vals, f"matched_delta_{metric}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"report": report,
                    "photometry_rows": photometry_rows,
                    "hough_rows": hough_rows,
                    "region_rows": region_rows}, f, indent=2)
    print(f"[{tag}] guardado en {out_json}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                         default=REPO_ROOT / "experiments" / "97-t2-crosshair-recheck")
    args = parser.parse_args()

    cache = args.out_dir / "cache"
    train_roots = [("Scenario_03", REPO_ROOT / "data" / "Task 2" / "Scenario_03")]
    for scenario_dir in sorted(cache.glob("Scenario_*")):
        train_roots.append((scenario_dir.name, scenario_dir))

    train_report = run(train_roots, args.out_dir / "results_train.json", "train")

    mock_root = REPO_ROOT / "data" / "Mock Test" / "Task 2" / "Scenario_12"
    mock_report = run([("Scenario_12_mock", mock_root)], args.out_dir / "results_mock.json", "mock")

    print("\n=== RESUMEN TRAIN ===")
    print(json.dumps(train_report, indent=2))
    print("\n=== RESUMEN MOCK ===")
    print(json.dumps(mock_report, indent=2))


if __name__ == "__main__":
    main()
