"""Deriva la composicion geometrica de la matriz GT de Task 2 a partir de las
poses 3D anotadas (T2-98, ver `experiments/98-t2-pose-estimation/HALLAZGO.md`),
la verifica con la metrica oficial (corner-AUC, leave-one-scenario-out) y mide
el listón de precision angular necesario.

Resumen de la derivacion (detalle completo en
`experiments/98-t2-pose-estimation/GEOMETRIA_EXACTA.md`):

1. La matriz GT (`Ground Truth.Task 2`) es EXACTAMENTE afin (fila 3 = [0,0,1]
   en los 1214 casos de train, verificado a precision de máquina) y
   aproximadamente una similitud reflejada (hallazgo previo, ver
   `src/fido/geometry.py::compose_similarity`), aunque NO exactamente: hay una
   asimetria residual de ~2.7 px de media entre A[0,1] y A[1,0] -- por eso el
   techo de un compositor de 4 DOF puro (rotacion+escala+traslacion) con GT
   perfecto es 0.6389 y no 1.0.

2. Solo `iOCT Microscope.Spatial.Rotation` explica la parte de ROTACION
   (`theta`) de la matriz de forma casi exacta: descomponiendo el cuaternion
   del iOCT en (tilt, azimuth, roll) respecto a un eje de mira fijo (calibrado
   sobre los datos: el eje local -Y del dispositivo, verificado porque es el
   que mas se acerca a apuntar hacia el origen del mundo -- angulo medio 8.7°,
   maximo 21.2°), se cumple EXACTAMENTE (residuo medio 0.78°, maximo 5.6° en
   1214 casos):

        theta_GT = -roll_iOCT + theta0         (theta0 = 0.06°, ~0)

   `Eyeball.Spatial.Rotation` NO aporta nada a `theta` (R² de un ajuste solo-
   ojo es <0.02 en los 4 parametros lineales de la matriz).

3. La ESCALA y la TRASLACION (posicion del centro del recuadro escaneado en
   la imagen) se explican de forma PARCIAL por una proyeccion gnomonica de
   primer orden del mismo eje de mira: con
   `gx = tan(tilt) cos(azimuth)`, `gy = tan(tilt) sin(azimuth)`,

        scale ~= s0 + s1*(gx^2 + gy^2)
        tx    ~= c_tx_x*gx + c_tx_y*gy + c_tx_0
        ty    ~= c_ty_x*gx + c_ty_y*gy + c_ty_0

   Esto deja un residuo de ~7-11 px (no despreciable dentro de la ventana de
   0-10 px de la metrica). Un intento de derivar la cadena fisica completa
   (rayo del iOCT -> interseccion con esfera de radio ajustable -> camara
   pinhole del Opmi con foco/centro ajustables, resuelto por minimos
   cuadrados no lineales con arranques multiples) NO convergio a un ajuste
   mejor (ver seccion "intento fallido" en el informe) -- se deja documentado
   como intento fallido, no como derivacion verificada.
   Anadir `Eyeball.Rotation` (aplanada, 9 numeros) a la regresion de
   traslacion mejora el ajuste EN MUESTRA pero empeora el AUC oficial bajo
   leave-one-scenario-out (sobreajuste) -- se descarta del modelo final.

Todos los numeros de este modulo son reproducibles ejecutando:
    python analysis/derive_task2_geometry.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fido.geometry import compose_similarity, corner_auc, corner_error  # noqa: E402

# --------------------------------------------------------------------------
# Constantes geometricas VERIFICADAS sobre los 1214 casos de train (no son
# libres: son la convencion de ejes locales del motor -- Unity-like, Y-up --
# confirmada por sanity check contra `Opmi.Spatial.Rotation`, que es fijo y
# conocido: [0.70711, 0, 0, 0.70711] = +90 grados en torno al eje X mundial).
# --------------------------------------------------------------------------
FWD_LOCAL_IOCT = np.array([0.0, -1.0, 0.0])   # eje de mira local del iOCT
RIGHT_LOCAL_IOCT = np.array([1.0, 0.0, 0.0])  # eje "derecha" local del iOCT


def quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """Cuaternion (..., 4) formato xyzw -> matriz de rotacion (..., 3, 3).

    Convencion verificada con `Opmi.Spatial.Rotation` = [0.70711,0,0,0.70711]:
    da una rotacion de +90 grados en torno a X (columna Y local -> Z mundial,
    columna Z local -> -Y mundial), consistente con un microscopio fijo justo
    encima del ojo (Opmi.Translation.Y = 96.7 > 0 = Eyeball.Translation.Y)
    mirando hacia abajo.
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    nrm = x * x + y * y + z * z + w * w
    s = 2.0 / nrm
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    R = np.zeros(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - (yy + zz); R[..., 0, 1] = xy - wz;     R[..., 0, 2] = xz + wy
    R[..., 1, 0] = xy + wz;       R[..., 1, 1] = 1 - (xx + zz); R[..., 1, 2] = yz - wx
    R[..., 2, 0] = xz - wy;       R[..., 2, 1] = yz + wx;     R[..., 2, 2] = 1 - (xx + yy)
    return R


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """xyzw * xyzw -> xyzw (misma convencion que `measure_task2_pose_estimation.py`)."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=1)


def random_small_rotation_quats(rng: np.random.Generator, n: int, angle_deg: float) -> np.ndarray:
    if angle_deg == 0.0:
        return np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = np.radians(angle_deg) / 2.0
    return np.concatenate([axis * np.sin(half), np.full((n, 1), np.cos(half))], axis=1)


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------

def load_dataset(annotations_root: Path) -> dict:
    files = sorted(annotations_root.glob("Scenario_*/*.json"))
    ioct_q, eye_q, mat, scenario = [], [], [], []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        ioct_q.append(data["iOCT Microscope"]["Spatial"]["Rotation"])
        eye_q.append(data["Eyeball"]["Spatial"]["Rotation"])
        mat.append(data["Ground Truth"]["Task 2"])
        scenario.append(f.parent.name.replace("Scenario_", ""))
    return dict(
        ioct_q=np.asarray(ioct_q, dtype=np.float64),
        eye_q=np.asarray(eye_q, dtype=np.float64),
        gt=np.asarray(mat, dtype=np.float64),
        scenario=np.asarray(scenario),
        case_id=np.array([f"{s}/{f.stem}" for s, f in zip(scenario, files)]),
    )


# --------------------------------------------------------------------------
# Features geometricas: (tilt, azimuth, roll) del cuaternion del iOCT
# respecto a un eje de mira `v0` calibrado sobre los datos de entrenamiento
# (la direccion promedio del eje local -Y del iOCT en el mundo -- constante
# fisica del rig, NO especifica de un caso; se recalibra por-fold en LOSO
# para no filtrar informacion del escenario de test).
# --------------------------------------------------------------------------

def calibrate_boresight(ioct_q_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = quat_to_rotmat_xyzw(ioct_q_train)
    v = np.einsum("nij,j->ni", R, FWD_LOCAL_IOCT)
    v0 = v.mean(axis=0)
    v0 /= np.linalg.norm(v0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(v0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, v0) * v0
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v0, e1)
    return v0, e1, e2


def geometric_features(ioct_q: np.ndarray, v0: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    R = quat_to_rotmat_xyzw(ioct_q)
    v = np.einsum("nij,j->ni", R, FWD_LOCAL_IOCT)
    r = np.einsum("nij,j->ni", R, RIGHT_LOCAL_IOCT)

    tilt = np.arccos(np.clip(v @ v0, -1.0, 1.0))
    azimuth = np.arctan2(v @ e2, v @ e1)
    gx = np.tan(tilt) * np.cos(azimuth)
    gy = np.tan(tilt) * np.sin(azimuth)

    ref = e1[None, :] - (e1 @ v.T)[:, None] * v
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    r_perp = r - np.sum(r * v, axis=1, keepdims=True) * v
    r_perp /= np.linalg.norm(r_perp, axis=1, keepdims=True)
    cos_roll = np.sum(r_perp * ref, axis=1)
    sin_roll = np.sum(np.cross(ref, r_perp) * v, axis=1)
    roll = np.arctan2(sin_roll, cos_roll)
    return gx, gy, roll, np.degrees(tilt)


def gt_similarity_params(gt: np.ndarray):
    a, b = gt[:, 0, 0], gt[:, 0, 1]
    tx, ty = gt[:, 0, 2], gt[:, 1, 2]
    scale = np.sqrt(a ** 2 + b ** 2)
    theta = np.arctan2(b / scale, a / scale)
    return theta, scale, tx, ty


# --------------------------------------------------------------------------
# Ajuste de los parametros libres (todo lineal en las features geometricas
# -> minimos cuadrados en forma cerrada, sin optimizacion iterativa)
# --------------------------------------------------------------------------

def fit_model(ioct_q_train, gt_train, v0, e1, e2):
    gx, gy, roll, _ = geometric_features(ioct_q_train, v0, e1, e2)
    theta_gt, scale_gt, tx_gt, ty_gt = gt_similarity_params(gt_train)
    n = len(ioct_q_train)

    # theta = -roll + theta0  (sistema lineal en (cos theta0, sin theta0))
    M = np.zeros((2 * n, 2))
    M[:n, 0], M[:n, 1] = np.cos(-roll), -np.sin(-roll)
    M[n:, 0], M[n:, 1] = np.sin(-roll), np.cos(-roll)
    y = np.concatenate([np.cos(theta_gt), np.sin(theta_gt)])
    sol, *_ = np.linalg.lstsq(M, y, rcond=None)
    theta0 = float(np.arctan2(sol[1], sol[0]))

    feat_scale = np.stack([np.ones(n), gx * gx + gy * gy], axis=1)
    coef_scale, *_ = np.linalg.lstsq(feat_scale, scale_gt, rcond=None)

    feat_xy = np.stack([gx, gy, np.ones(n)], axis=1)
    coef_tx, *_ = np.linalg.lstsq(feat_xy, tx_gt, rcond=None)
    coef_ty, *_ = np.linalg.lstsq(feat_xy, ty_gt, rcond=None)

    return dict(theta0=theta0, coef_scale=coef_scale, coef_tx=coef_tx, coef_ty=coef_ty)


def predict_matrix(ioct_q, v0, e1, e2, params):
    gx, gy, roll, _ = geometric_features(ioct_q, v0, e1, e2)
    theta = -roll + params["theta0"]
    scale = params["coef_scale"][0] + params["coef_scale"][1] * (gx * gx + gy * gy)
    tx = params["coef_tx"][0] * gx + params["coef_tx"][1] * gy + params["coef_tx"][2]
    ty = params["coef_ty"][0] * gx + params["coef_ty"][1] * gy + params["coef_ty"][2]
    return compose_similarity(tx, ty, np.cos(theta), np.sin(theta), scale, reflect=True)


# --------------------------------------------------------------------------
# Evaluacion leave-one-scenario-out con la metrica oficial
# --------------------------------------------------------------------------

def run_loso(data: dict) -> dict:
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    n = len(ioct_q)
    all_errors = np.zeros(n)
    per_scenario = {}
    for scen in np.unique(scenario):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        v0, e1, e2 = calibrate_boresight(ioct_q[train_idx])
        params = fit_model(ioct_q[train_idx], gt[train_idx], v0, e1, e2)
        pred = predict_matrix(ioct_q[test_idx], v0, e1, e2, params)
        errs = corner_error(pred, gt[test_idx])
        all_errors[test_idx] = errs
        per_scenario[scen] = dict(n=int(len(test_idx)), mean_error=float(errs.mean()),
                                   median_error=float(np.median(errs)), auc=float(corner_auc(errs)))
    return dict(per_scenario=per_scenario, pooled_auc=float(corner_auc(all_errors)),
                pooled_mean_error=float(all_errors.mean()),
                pooled_median_error=float(np.median(all_errors)),
                frac_under_10px=float((all_errors <= 10).mean()))


def run_sensitivity_sweep(data: dict, angles_deg: list[float], trials: int, seed: int) -> list[dict]:
    ioct_q, eye_q, gt = data["ioct_q"], data["eye_q"], data["gt"]
    n = len(ioct_q)
    v0, e1, e2 = calibrate_boresight(ioct_q)          # calibracion global (constante de rig)
    params = fit_model(ioct_q, gt, v0, e1, e2)         # ajuste sobre TODO train (para el barrido)
    rng = np.random.default_rng(seed)
    rows = []
    for angle in angles_deg:
        for perturb in ("ioct_only", "eye_only", "both"):
            aucs = []
            for _ in range(trials):
                if angle == 0.0 or perturb == "eye_only":
                    q_noisy = ioct_q  # el modelo no usa eye_q -> perturbarlo solo no cambia nada
                else:
                    noise = random_small_rotation_quats(rng, n, angle)
                    q_noisy = quat_multiply(noise, ioct_q)
                pred = predict_matrix(q_noisy, v0, e1, e2, params)
                errs = corner_error(pred, gt)
                aucs.append(corner_auc(errs))
            rows.append(dict(angle_deg=angle, perturb=perturb,
                              auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs))))
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path,
                        default=REPO_ROOT / "data/_annotations/Task 2")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "experiments/98-t2-pose-estimation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_dataset(args.annotations_root)
    n = len(data["ioct_q"])
    print(f"n_cases={n}  scenarios={sorted(set(data['scenario']))}")

    # --- Parametros finales (ajustados sobre TODO train, para reportar) ---
    v0, e1, e2 = calibrate_boresight(data["ioct_q"])
    params = fit_model(data["ioct_q"], data["gt"], v0, e1, e2)
    print("\n=== Parametros ajustados (todo train, n=%d) ===" % n)
    print(f"v0 (eje de mira, marco mundo)        = {np.round(v0, 5).tolist()}")
    print(f"e1 (referencia de azimuth, marco mundo) = {np.round(e1, 5).tolist()}")
    print(f"theta0 = {np.degrees(params['theta0']):.4f} deg")
    print(f"scale: s0={params['coef_scale'][0]:.4f}  s1={params['coef_scale'][1]:.4f}")
    print(f"tx: c_gx={params['coef_tx'][0]:.4f} c_gy={params['coef_tx'][1]:.4f} c0={params['coef_tx'][2]:.4f}")
    print(f"ty: c_gx={params['coef_ty'][0]:.4f} c_gy={params['coef_ty'][1]:.4f} c0={params['coef_ty'][2]:.4f}")

    pred_insample = predict_matrix(data["ioct_q"], v0, e1, e2, params)
    err_insample = corner_error(pred_insample, data["gt"])
    print(f"\nAUC en muestra (ajuste y evaluacion sobre los mismos 1214 casos, "
          f"referencia optimista, NO es el resultado que cuenta): "
          f"{corner_auc(err_insample):.4f}  mean_err={err_insample.mean():.2f}px")

    theta_gt, scale_gt, tx_gt, ty_gt = gt_similarity_params(data["gt"])
    gx, gy, roll, tilt_deg = geometric_features(data["ioct_q"], v0, e1, e2)
    theta_pred = -roll + params["theta0"]
    theta_resid_deg = np.degrees(np.abs(np.angle(np.exp(1j * (theta_pred - theta_gt)))))
    print(f"\nResiduo de theta (rotacion) = -roll_iOCT + theta0: "
          f"media={theta_resid_deg.mean():.3f} deg, max={theta_resid_deg.max():.3f} deg "
          "[relacion CASI EXACTA]")
    print(f"tilt del eje de mira: media={tilt_deg.mean():.2f} deg, max={tilt_deg.max():.2f} deg")

    # --- Evaluacion oficial: leave-one-scenario-out ---
    print("\n=== Leave-one-scenario-out (metrica oficial) ===")
    loso = run_loso(data)
    for scen, stats in sorted(loso["per_scenario"].items()):
        print(f"  scenario {scen}: n={stats['n']:4d} mean_err={stats['mean_error']:6.2f}px "
              f"median={stats['median_error']:6.2f}px AUC={stats['auc']:.4f}")
    print(f"POOLED: AUC={loso['pooled_auc']:.4f}  mean_err={loso['pooled_mean_error']:.2f}px "
          f"median={loso['pooled_median_error']:.2f}px  frac<10px={loso['frac_under_10px']:.4f}")
    print("Referencia: modelo lineal (regresion sobre 18 componentes de cuaterniones) "
          "AUC=0.0937 | techo con matriz GT reproyectada como similitud 4-DOF AUC=0.6389")

    # --- Barrido de precision angular ---
    print("\n=== Barrido de precision angular ===")
    angles = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    sweep = run_sensitivity_sweep(data, angles, trials=8, seed=args.seed)
    for row in sweep:
        print(f"  angulo={row['angle_deg']:5.2f}deg  perturb={row['perturb']:10s} "
              f"AUC={row['auc_mean']:.4f} +- {row['auc_std']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = dict(
        n_cases=n,
        boresight_v0=v0.tolist(), azimuth_ref_e1=e1.tolist(),
        theta0_deg=float(np.degrees(params["theta0"])),
        coef_scale=params["coef_scale"].tolist(),
        coef_tx=params["coef_tx"].tolist(), coef_ty=params["coef_ty"].tolist(),
        theta_residual_deg=dict(mean=float(theta_resid_deg.mean()), max=float(theta_resid_deg.max())),
        tilt_deg=dict(mean=float(tilt_deg.mean()), max=float(tilt_deg.max())),
        insample_auc=float(corner_auc(err_insample)),
        insample_mean_error_px=float(err_insample.mean()),
        loso=loso,
        baseline_linear_auc=0.0937,
        ceiling_similarity4dof_auc=0.6389,
        sensitivity_sweep=sweep,
    )
    out_path = args.output_dir / "derive_task2_geometry_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
