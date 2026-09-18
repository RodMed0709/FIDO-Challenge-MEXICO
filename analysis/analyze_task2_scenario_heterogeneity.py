"""T2-99 -- por que el modelo geometrico de Task 2 (derive_task2_geometry.py)
funciona en unos escenarios y falla por completo en otros.

Contexto (ver experiments/98-t2-pose-estimation/GEOMETRIA_EXACTA.md, seccion 4):
el modelo geometrico (theta = -roll_iOCT + theta0; escala/traslacion via
gnomonico de 1er orden de (tilt, azimuth)) da AUC oficial LOSO pooled=0.1512,
pero el desglose por escenario va de 0.0000 (02/05/10) a ~0.68-0.69 (07/08) --
por ENCIMA del lider del leaderboard (0.475).

Este script:
  1. Reproduce el desglose por escenario (llamando directamente a las
     funciones de derive_task2_geometry.py, sin reimplementar el modelo).
  2. Comprueba, con numeros, cuatro hipotesis sobre que distingue escenarios
     buenos de malos: (a) rango angular de tilt del iOCT respecto al eje de
     mira, (b) escala GT, (c) residuo de la relacion de rotacion theta, (d)
     CUALQUIER campo de la anotacion cruda que sea ~constante DENTRO de cada
     escenario pero distinto ENTRE escenarios (candidato a parametro fijo del
     simulador que en realidad varia por escenario).
  3. Si aparece una variable ganadora, prueba calibrar el modelo POR ESCENARIO
     (o anadir esa variable como feature) y remide el AUC LOSO.

Todo numpy/scipy, local, sin GPU. Ejecutar:
    python analysis/analyze_task2_scenario_heterogeneity.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fido.geometry import compose_similarity, corner_auc, corner_error  # noqa: E402

# --------------------------------------------------------------------------
# Importa derive_task2_geometry.py por ruta (no es un paquete instalable) para
# reusar EXACTAMENTE las mismas funciones que produjeron los numeros de
# GEOMETRIA_EXACTA.md -- cero reimplementacion, cero riesgo de divergencia.
# --------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "derive_task2_geometry", REPO_ROOT / "analysis" / "derive_task2_geometry.py"
)
dtg = importlib.util.module_from_spec(_spec)
sys.modules["derive_task2_geometry"] = dtg
_spec.loader.exec_module(dtg)

ANNOTATIONS_ROOT = REPO_ROOT / "data" / "_annotations" / "Task 2"
OUTPUT_DIR = REPO_ROOT / "experiments" / "99-t2-scenario-heterogeneity"


# --------------------------------------------------------------------------
# 1. Carga de datos: reusa load_dataset pero ademas guarda TODOS los campos
#    crudos de la anotacion (no solo iOCT.Rotation / GT) para la busqueda de
#    la hipotesis (d): un campo con sd=0 global pero distinto por escenario.
# --------------------------------------------------------------------------

RAW_FIELD_PATHS = {
    "Opmi.Translation": ("Opmi", "Spatial", "Translation"),
    "Opmi.Rotation": ("Opmi", "Spatial", "Rotation"),
    "iOCT.Translation": ("iOCT Microscope", "Spatial", "Translation"),
    "iOCT.Rotation": ("iOCT Microscope", "Spatial", "Rotation"),
    "Eyeball.Translation": ("Eyeball", "Spatial", "Translation"),
    "Eyeball.Rotation": ("Eyeball", "Spatial", "Rotation"),
}


def load_full_dataset(root: Path) -> dict:
    files = sorted(root.glob("Scenario_*/*.json"))
    out = {k: [] for k in RAW_FIELD_PATHS}
    ioct_q, eye_q, mat, scenario, case_id = [], [], [], [], []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for name, path in RAW_FIELD_PATHS.items():
            node = data
            for key in path:
                node = node[key]
            out[name].append(node)
        ioct_q.append(data["iOCT Microscope"]["Spatial"]["Rotation"])
        eye_q.append(data["Eyeball"]["Spatial"]["Rotation"])
        mat.append(data["Ground Truth"]["Task 2"])
        scenario.append(f.parent.name.replace("Scenario_", ""))
        case_id.append(f"{f.parent.name}/{f.stem}")
    result = dict(
        ioct_q=np.asarray(ioct_q, dtype=np.float64),
        eye_q=np.asarray(eye_q, dtype=np.float64),
        gt=np.asarray(mat, dtype=np.float64),
        scenario=np.asarray(scenario),
        case_id=np.array(case_id),
    )
    for name in RAW_FIELD_PATHS:
        result[f"raw__{name}"] = np.asarray(out[name], dtype=np.float64)
    return result


# --------------------------------------------------------------------------
# 2a. Hipotesis (a): rango de tilt del iOCT respecto al eje de mira, por
#     escenario, correlacionado con el AUC del escenario.
# --------------------------------------------------------------------------

def hypothesis_tilt_range(data: dict, loso: dict) -> dict:
    v0, e1, e2 = dtg.calibrate_boresight(data["ioct_q"])  # calibracion global, solo descriptiva
    gx, gy, roll, tilt_deg = dtg.geometric_features(data["ioct_q"], v0, e1, e2)
    scenario = data["scenario"]
    rows = []
    for scen in sorted(set(scenario)):
        idx = scenario == scen
        t = tilt_deg[idx]
        rows.append(dict(
            scenario=scen, n=int(idx.sum()),
            tilt_mean=float(t.mean()), tilt_median=float(np.median(t)),
            tilt_min=float(t.min()), tilt_max=float(t.max()), tilt_std=float(t.std()),
            auc=loso["per_scenario"][scen]["auc"],
            mean_error=loso["per_scenario"][scen]["mean_error"],
        ))
    tilt_mean_arr = np.array([r["tilt_mean"] for r in rows])
    tilt_max_arr = np.array([r["tilt_max"] for r in rows])
    auc_arr = np.array([r["auc"] for r in rows])
    corr_mean = float(np.corrcoef(tilt_mean_arr, auc_arr)[0, 1])
    corr_max = float(np.corrcoef(tilt_max_arr, auc_arr)[0, 1])
    return dict(rows=rows, corr_tilt_mean_vs_auc=corr_mean, corr_tilt_max_vs_auc=corr_max)


# --------------------------------------------------------------------------
# 2b. Hipotesis (b): escala GT por escenario, correlacionada con AUC.
# --------------------------------------------------------------------------

def hypothesis_gt_scale(data: dict, loso: dict) -> dict:
    theta_gt, scale_gt, tx_gt, ty_gt = dtg.gt_similarity_params(data["gt"])
    scenario = data["scenario"]
    rows = []
    for scen in sorted(set(scenario)):
        idx = scenario == scen
        s = scale_gt[idx]
        rows.append(dict(
            scenario=scen, n=int(idx.sum()),
            scale_mean=float(s.mean()), scale_std=float(s.std()),
            scale_min=float(s.min()), scale_max=float(s.max()),
            auc=loso["per_scenario"][scen]["auc"],
        ))
    scale_mean_arr = np.array([r["scale_mean"] for r in rows])
    auc_arr = np.array([r["auc"] for r in rows])
    corr = float(np.corrcoef(scale_mean_arr, auc_arr)[0, 1])
    return dict(rows=rows, corr_scale_mean_vs_auc=corr)


# --------------------------------------------------------------------------
# 2c. Hipotesis (c): residuo de la relacion theta = -roll + theta0, por
#     escenario (calibrado GLOBALMENTE, como en GEOMETRIA_EXACTA.md 1.4, para
#     que sea un residuo puramente descriptivo -- no el LOSO por-fold).
# --------------------------------------------------------------------------

def hypothesis_theta_residual(data: dict, loso: dict) -> dict:
    v0, e1, e2 = dtg.calibrate_boresight(data["ioct_q"])
    params = dtg.fit_model(data["ioct_q"], data["gt"], v0, e1, e2)
    gx, gy, roll, tilt_deg = dtg.geometric_features(data["ioct_q"], v0, e1, e2)
    theta_gt, scale_gt, tx_gt, ty_gt = dtg.gt_similarity_params(data["gt"])
    theta_pred = -roll + params["theta0"]
    resid_deg = np.degrees(np.abs(np.angle(np.exp(1j * (theta_pred - theta_gt)))))
    scenario = data["scenario"]
    rows = []
    for scen in sorted(set(scenario)):
        idx = scenario == scen
        r = resid_deg[idx]
        rows.append(dict(
            scenario=scen, n=int(idx.sum()),
            resid_mean=float(r.mean()), resid_median=float(np.median(r)), resid_max=float(r.max()),
            auc=loso["per_scenario"][scen]["auc"],
        ))
    resid_mean_arr = np.array([r["resid_mean"] for r in rows])
    auc_arr = np.array([r["auc"] for r in rows])
    corr = float(np.corrcoef(resid_mean_arr, auc_arr)[0, 1])
    return dict(rows=rows, corr_resid_mean_vs_auc=corr)


# --------------------------------------------------------------------------
# 2d. Hipotesis (d): TODOS los campos crudos, agrupados por escenario. Busca
#     campos con sd~0 DENTRO de cada escenario (constante por escenario) pero
#     media claramente distinta ENTRE escenarios -- candidato a parametro del
#     simulador que el modelo actual asume constante y no lo es.
# --------------------------------------------------------------------------

def hypothesis_hidden_scenario_param(data: dict, loso: dict) -> dict:
    scenario = data["scenario"]
    scenarios_sorted = sorted(set(scenario))
    auc_by_scenario = np.array([loso["per_scenario"][s]["auc"] for s in scenarios_sorted])

    findings = []
    for field_name in RAW_FIELD_PATHS:
        arr = data[f"raw__{field_name}"]  # (n, d)
        d = arr.shape[1]
        for comp in range(d):
            vals = arr[:, comp]
            global_sd = float(vals.std())
            per_scenario_means = []
            per_scenario_sds = []
            for scen in scenarios_sorted:
                idx = scenario == scen
                per_scenario_means.append(float(vals[idx].mean()))
                per_scenario_sds.append(float(vals[idx].std()))
            per_scenario_means = np.array(per_scenario_means)
            per_scenario_sds = np.array(per_scenario_sds)
            max_within_sd = float(per_scenario_sds.max())
            between_scenario_sd = float(per_scenario_means.std())
            # Candidato interesante: variacion ENTRE escenarios notablemente
            # mayor que la variacion DENTRO de cada escenario (constante por
            # escenario, pero NO globalmente constante).
            ratio = between_scenario_sd / (max_within_sd + 1e-12)
            corr_auc = float(np.corrcoef(per_scenario_means, auc_by_scenario)[0, 1]) \
                if between_scenario_sd > 1e-9 else 0.0
            findings.append(dict(
                field=f"{field_name}[{comp}]",
                global_sd=global_sd,
                max_within_scenario_sd=max_within_sd,
                between_scenario_sd=between_scenario_sd,
                ratio_between_over_within=ratio,
                corr_with_auc=corr_auc,
                per_scenario_means=per_scenario_means.tolist(),
            ))
    findings.sort(key=lambda r: -r["ratio_between_over_within"])
    return dict(scenarios=scenarios_sorted, findings=findings)


# --------------------------------------------------------------------------
# 3. Si hay variable ganadora: recalibrar POR ESCENARIO (oraculo, usa el
#    escenario de test para elegir su propio ajuste -- SOLO para medir el
#    techo de lo que "saber el escenario" aportaria, no un modelo desplegable)
#    y comparar contra LOSO real.
# --------------------------------------------------------------------------

def per_scenario_oracle_calibration(data: dict) -> dict:
    """Ajusta v0/e1/e2 y los coeficientes del modelo SOLO con los datos del
    propio escenario de test (n pequeno, ~65-183 casos) -- mide si la falla es
    de CALIBRACION del modelo (arreglable por escenario) o de FORMA FUNCIONAL
    (el gnomonico de 1er orden no alcanza aunque se calibre localmente)."""
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    rows = {}
    all_errors = np.zeros(len(ioct_q))
    for scen in sorted(set(scenario)):
        idx = np.where(scenario == scen)[0]
        v0, e1, e2 = dtg.calibrate_boresight(ioct_q[idx])
        params = dtg.fit_model(ioct_q[idx], gt[idx], v0, e1, e2)
        pred = dtg.predict_matrix(ioct_q[idx], v0, e1, e2, params)
        errs = corner_error(pred, gt[idx])
        all_errors[idx] = errs
        rows[scen] = dict(n=int(len(idx)), mean_error=float(errs.mean()),
                           median_error=float(np.median(errs)), auc=float(corner_auc(errs)))
    return dict(per_scenario=rows, pooled_auc=float(corner_auc(all_errors)),
                pooled_mean_error=float(all_errors.mean()))


def boresight_mismatch_vs_bias(data: dict, loso: dict) -> dict:
    """Para cada fold LOSO: angulo entre el eje de mira v0 ajustado con los 9
    escenarios de train y el ajustado SOLO con el escenario de test (oraculo,
    unicamente para medir el desajuste, no para predecir); lo compara con el
    sesgo (bias, no dispersion) que deja el modelo LOSO real en scale/tx/ty
    para ese escenario. Aisla si el fallo es del EJE o de los COEFICIENTES."""
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    rows = []
    for scen in sorted(set(scenario)):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        v0_tr, e1_tr, e2_tr = dtg.calibrate_boresight(ioct_q[train_idx])
        v0_te, _, _ = dtg.calibrate_boresight(ioct_q[test_idx])
        boresight_err_deg = float(np.degrees(np.arccos(np.clip(np.dot(v0_tr, v0_te), -1, 1))))

        params = dtg.fit_model(ioct_q[train_idx], gt[train_idx], v0_tr, e1_tr, e2_tr)
        gx, gy, roll, _ = dtg.geometric_features(ioct_q[test_idx], v0_tr, e1_tr, e2_tr)
        scale_pred = params["coef_scale"][0] + params["coef_scale"][1] * (gx * gx + gy * gy)
        tx_pred = params["coef_tx"][0] * gx + params["coef_tx"][1] * gy + params["coef_tx"][2]
        ty_pred = params["coef_ty"][0] * gx + params["coef_ty"][1] * gy + params["coef_ty"][2]
        theta_gt, scale_gt, tx_gt, ty_gt = dtg.gt_similarity_params(gt[test_idx])
        scale_bias = float((scale_pred - scale_gt).mean())
        tx_bias = float((tx_pred - tx_gt).mean())
        ty_bias = float((ty_pred - ty_gt).mean())
        trans_bias_mag = float(np.hypot(tx_bias, ty_bias))
        rows.append(dict(scenario=scen, boresight_err_deg=boresight_err_deg,
                          scale_bias_px=scale_bias, trans_bias_px=trans_bias_mag,
                          auc=loso["per_scenario"][scen]["auc"]))
    angs = np.array([r["boresight_err_deg"] for r in rows])
    scale_b = np.abs(np.array([r["scale_bias_px"] for r in rows]))
    trans_b = np.array([r["trans_bias_px"] for r in rows])
    aucs = np.array([r["auc"] for r in rows])
    return dict(
        rows=rows,
        corr_boresight_err_vs_scale_bias=float(np.corrcoef(angs, scale_b)[0, 1]),
        corr_boresight_err_vs_trans_bias=float(np.corrcoef(angs, trans_b)[0, 1]),
        corr_scale_bias_vs_auc=float(np.corrcoef(scale_b, aucs)[0, 1]),
        corr_trans_bias_vs_auc=float(np.corrcoef(trans_b, aucs)[0, 1]),
    )


def coefficient_transfer_ablations(data: dict) -> dict:
    """Ablaciones que aislan QUE parte del modelo no transfiere entre
    escenarios bajo LOSO: el eje de mira (v0/e1/e2) o los coeficientes
    (theta0, coef_scale, coef_tx, coef_ty). Para cada ablacion, una o mas
    piezas del modelo se sustituyen por un ORACULO ajustado SOLO con el
    escenario de test (diagnostico -- no es un modelo desplegable, ya que usa
    el GT del propio escenario que se evalua)."""
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    n = len(ioct_q)

    def run(which: tuple, oracle_axis: bool) -> tuple:
        all_err = np.zeros(n)
        per_scen = {}
        for scen in sorted(set(scenario)):
            test_idx = np.where(scenario == scen)[0]
            train_idx = np.where(scenario != scen)[0]
            v0_train, e1_train, e2_train = dtg.calibrate_boresight(ioct_q[train_idx])
            if oracle_axis:
                v0, e1, e2 = dtg.calibrate_boresight(ioct_q[test_idx])
            else:
                v0, e1, e2 = v0_train, e1_train, e2_train
            params_train = dtg.fit_model(ioct_q[train_idx], gt[train_idx], v0, e1, e2)
            params_oracle = dtg.fit_model(ioct_q[test_idx], gt[test_idx], v0, e1, e2)
            params = dict(params_train)
            for key, pname in (("theta0", "theta0"), ("scale", "coef_scale"),
                                ("tx", "coef_tx"), ("ty", "coef_ty")):
                if key in which:
                    params[pname] = params_oracle[pname]
            pred = dtg.predict_matrix(ioct_q[test_idx], v0, e1, e2, params)
            errs = corner_error(pred, gt[test_idx])
            all_err[test_idx] = errs
            per_scen[scen] = float(corner_auc(errs))
        return float(corner_auc(all_err)), float(all_err.mean()), per_scen

    ablations = {}
    # V: solo el eje viene del oraculo, coeficientes de los otros 9 (relativos a ese eje)
    auc, me, per = run((), oracle_axis=True)
    ablations["axis_only"] = dict(auc=auc, mean_error=me, per_scenario=per)
    # C: eje de los otros 9 (como LOSO real), TODOS los coeficientes del oraculo
    auc, me, per = run(("theta0", "scale", "tx", "ty"), oracle_axis=False)
    ablations["all_coefficients_only"] = dict(auc=auc, mean_error=me, per_scenario=per)
    # Descomposicion por coeficiente individual (eje real de LOSO)
    for which in [("theta0",), ("scale",), ("tx", "ty"), ("tx",), ("ty",),
                  ("scale", "tx"), ("scale", "ty"), ("scale", "tx", "ty")]:
        auc, me, per = run(which, oracle_axis=False)
        ablations["+".join(which)] = dict(auc=auc, mean_error=me, per_scenario=per)
    # Referencia: oraculo completo (eje + todos los coeficientes, del propio escenario)
    auc, me, per = run(("theta0", "scale", "tx", "ty"), oracle_axis=True)
    ablations["full_oracle"] = dict(auc=auc, mean_error=me, per_scenario=per)
    return ablations


def eye_rotation_scenario_correlation(data: dict) -> dict:
    """Explora si la MEDIA por escenario de algun componente de
    Eyeball.Rotation predice el intercepto de escala (s0) que el oraculo por
    escenario encuentra -- posible pista fisica de por que theta0/scale/tx/ty
    no transfieren, au que hipotesis (d) ya descarto que sea un campo
    'constante por escenario' en sentido estricto (dispersion intra-escenario
    grande). Se reporta con su p-valor: n=10 escenarios, riesgo alto de
    correlacion espuria con comparaciones multiples."""
    from scipy import stats as sstats
    ioct_q, eye_q, gt, scenario = data["ioct_q"], data["eye_q"], data["gt"], data["scenario"]
    v0, e1, e2 = dtg.calibrate_boresight(ioct_q)
    scens = sorted(set(scenario))
    s0_oracle, eye_comp_means, within_sds = [], [], []
    for scen in scens:
        idx = scenario == scen
        params = dtg.fit_model(ioct_q[idx], gt[idx], v0, e1, e2)
        s0_oracle.append(params["coef_scale"][0])
        eye_comp_means.append(eye_q[idx, 2].mean())  # componente z del cuaternion del ojo
        within_sds.append(float(eye_q[idx, 2].std()))
    s0_oracle = np.asarray(s0_oracle)
    eye_comp_means = np.asarray(eye_comp_means)
    r, p = sstats.pearsonr(eye_comp_means, s0_oracle)
    return dict(
        component="Eyeball.Rotation[z]",
        per_scenario_mean=dict(zip(scens, eye_comp_means.tolist())),
        s0_oracle_per_scenario=dict(zip(scens, s0_oracle.tolist())),
        corr_r=float(r), corr_p=float(p),
        between_scenario_sd_of_means=float(eye_comp_means.std()),
        max_within_scenario_sd=float(max(within_sds)),
        note="within_sd > between_sd => la 'senal' es mas ruido intra-escenario que diferencia real entre escenarios",
    )


def per_scenario_oracle_tilt_bucket(data: dict, loso: dict) -> dict:
    """Si el tilt es la variable ganadora: en vez de calibrar TODO por
    escenario (que requiere conocer el escenario en test), prueba anadir el
    tilt/tilt^2 como feature EXTRA en el ajuste global de escala/traslacion
    (mismo LOSO real, sin oraculo de escenario) -- mide si extender el orden
    del modelo (no el escenario en si) resuelve el problema."""
    ioct_q, gt, scenario = data["ioct_q"], data["gt"], data["scenario"]
    n = len(ioct_q)
    all_errors = np.zeros(n)
    per_scenario = {}
    for scen in sorted(set(scenario)):
        test_idx = np.where(scenario == scen)[0]
        train_idx = np.where(scenario != scen)[0]
        v0, e1, e2 = dtg.calibrate_boresight(ioct_q[train_idx])
        gx_tr, gy_tr, roll_tr, _ = dtg.geometric_features(ioct_q[train_idx], v0, e1, e2)
        theta_gt_tr, scale_gt_tr, tx_gt_tr, ty_gt_tr = dtg.gt_similarity_params(gt[train_idx])
        ntr = len(train_idx)

        # theta: identico al modelo base
        M = np.zeros((2 * ntr, 2))
        M[:ntr, 0], M[:ntr, 1] = np.cos(-roll_tr), -np.sin(-roll_tr)
        M[ntr:, 0], M[ntr:, 1] = np.sin(-roll_tr), np.cos(-roll_tr)
        y = np.concatenate([np.cos(theta_gt_tr), np.sin(theta_gt_tr)])
        sol, *_ = np.linalg.lstsq(M, y, rcond=None)
        theta0 = float(np.arctan2(sol[1], sol[0]))

        # escala/tx/ty: features extendidas con gx^3,gy^3, gx*gy, (gx^2+gy^2)^2
        r2 = gx_tr * gx_tr + gy_tr * gy_tr
        feat_scale = np.stack([np.ones(ntr), r2, r2 * r2], axis=1)
        coef_scale, *_ = np.linalg.lstsq(feat_scale, scale_gt_tr, rcond=None)
        feat_xy = np.stack([gx_tr, gy_tr, gx_tr * gx_tr, gy_tr * gy_tr, gx_tr * gy_tr,
                             np.ones(ntr)], axis=1)
        coef_tx, *_ = np.linalg.lstsq(feat_xy, tx_gt_tr, rcond=None)
        coef_ty, *_ = np.linalg.lstsq(feat_xy, ty_gt_tr, rcond=None)

        gx_te, gy_te, roll_te, _ = dtg.geometric_features(ioct_q[test_idx], v0, e1, e2)
        theta_te = -roll_te + theta0
        r2_te = gx_te * gx_te + gy_te * gy_te
        scale_te = coef_scale[0] + coef_scale[1] * r2_te + coef_scale[2] * r2_te * r2_te
        feat_xy_te = np.stack([gx_te, gy_te, gx_te * gx_te, gy_te * gy_te, gx_te * gy_te,
                                np.ones(len(test_idx))], axis=1)
        tx_te = feat_xy_te @ coef_tx
        ty_te = feat_xy_te @ coef_ty
        pred = compose_similarity(tx_te, ty_te, np.cos(theta_te), np.sin(theta_te), scale_te, reflect=True)
        errs = corner_error(pred, gt[test_idx])
        all_errors[test_idx] = errs
        per_scenario[scen] = dict(n=int(len(test_idx)), mean_error=float(errs.mean()),
                                   median_error=float(np.median(errs)), auc=float(corner_auc(errs)))
    return dict(per_scenario=per_scenario, pooled_auc=float(corner_auc(all_errors)),
                pooled_mean_error=float(all_errors.mean()))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    data = load_full_dataset(ANNOTATIONS_ROOT)
    n = len(data["ioct_q"])
    print(f"n_cases={n}  scenarios={sorted(set(data['scenario']))}")

    print("\n=== 1. Desglose LOSO (reproduccion exacta de GEOMETRIA_EXACTA.md sec.4) ===")
    loso = dtg.run_loso(data)
    order = sorted(loso["per_scenario"].items(), key=lambda kv: -kv[1]["auc"])
    for scen, stats in order:
        print(f"  scenario {scen}: n={stats['n']:4d} mean_err={stats['mean_error']:6.2f}px "
              f"median={stats['median_error']:6.2f}px AUC={stats['auc']:.4f}")
    print(f"POOLED: AUC={loso['pooled_auc']:.4f} mean_err={loso['pooled_mean_error']:.2f}px")

    print("\n=== 2a. Hipotesis: rango de tilt del iOCT por escenario ===")
    h_tilt = hypothesis_tilt_range(data, loso)
    for r in sorted(h_tilt["rows"], key=lambda r: -r["auc"]):
        print(f"  scen {r['scenario']}: tilt mean={r['tilt_mean']:5.2f} "
              f"[{r['tilt_min']:5.2f},{r['tilt_max']:5.2f}] std={r['tilt_std']:4.2f} "
              f"-> AUC={r['auc']:.4f}")
    print(f"  corr(tilt_mean, AUC) = {h_tilt['corr_tilt_mean_vs_auc']:.3f}")
    print(f"  corr(tilt_max,  AUC) = {h_tilt['corr_tilt_max_vs_auc']:.3f}")

    print("\n=== 2b. Hipotesis: escala GT por escenario ===")
    h_scale = hypothesis_gt_scale(data, loso)
    for r in sorted(h_scale["rows"], key=lambda r: -r["auc"]):
        print(f"  scen {r['scenario']}: scale mean={r['scale_mean']:6.2f} "
              f"[{r['scale_min']:6.2f},{r['scale_max']:6.2f}] -> AUC={r['auc']:.4f}")
    print(f"  corr(scale_mean, AUC) = {h_scale['corr_scale_mean_vs_auc']:.3f}")

    print("\n=== 2c. Hipotesis: residuo de theta=-roll+theta0 por escenario ===")
    h_resid = hypothesis_theta_residual(data, loso)
    for r in sorted(h_resid["rows"], key=lambda r: -r["auc"]):
        print(f"  scen {r['scenario']}: resid mean={r['resid_mean']:5.2f} "
              f"median={r['resid_median']:5.2f} max={r['resid_max']:5.2f} -> AUC={r['auc']:.4f}")
    print(f"  corr(resid_mean, AUC) = {h_resid['corr_resid_mean_vs_auc']:.3f}")

    print("\n=== 2d. Hipotesis: parametro oculto constante-por-escenario ===")
    h_hidden = hypothesis_hidden_scenario_param(data, loso)
    print("  top 12 campos por ratio (variacion ENTRE escenarios / variacion DENTRO de escenario):")
    for f in h_hidden["findings"][:12]:
        print(f"    {f['field']:<24} global_sd={f['global_sd']:9.4f} "
              f"within_sd_max={f['max_within_scenario_sd']:9.4f} "
              f"between_sd={f['between_scenario_sd']:9.4f} "
              f"ratio={f['ratio_between_over_within']:9.1f} "
              f"corr_auc={f['corr_with_auc']:+.3f}")

    print("\n=== 2e. Mecanismo: desajuste del eje de mira vs sesgo de scale/tx/ty (por fold LOSO) ===")
    h_mech = boresight_mismatch_vs_bias(data, loso)
    for r in sorted(h_mech["rows"], key=lambda r: -r["auc"]):
        print(f"  scen {r['scenario']}: boresight_err={r['boresight_err_deg']:6.3f}deg "
              f"scale_bias={r['scale_bias_px']:+7.2f}px trans_bias={r['trans_bias_px']:6.2f}px "
              f"AUC={r['auc']:.4f}")
    print(f"  corr(boresight_err, |scale_bias|) = {h_mech['corr_boresight_err_vs_scale_bias']:+.3f}")
    print(f"  corr(boresight_err, |trans_bias|) = {h_mech['corr_boresight_err_vs_trans_bias']:+.3f}")
    print(f"  corr(|scale_bias|, AUC) = {h_mech['corr_scale_bias_vs_auc']:+.3f}")
    print(f"  corr(|trans_bias|, AUC) = {h_mech['corr_trans_bias_vs_auc']:+.3f}")

    print("\n=== 2f. Ablaciones oraculo: eje vs coeficientes (aisla la causa) ===")
    ablations = coefficient_transfer_ablations(data)
    for name, res in ablations.items():
        print(f"  {name:<28s} AUC={res['auc']:.4f} mean_err={res['mean_error']:6.2f}px")

    print("\n=== 2g. Eyeball.Rotation[z] por escenario vs intercepto de escala (s0) del oraculo ===")
    h_eye = eye_rotation_scenario_correlation(data)
    print(f"  corr(mean eye_z, s0_oracle) r={h_eye['corr_r']:+.3f} p={h_eye['corr_p']:.4f}  "
          f"(between_sd={h_eye['between_scenario_sd_of_means']:.4f} vs "
          f"within_sd_max={h_eye['max_within_scenario_sd']:.4f})")

    print("\n=== 3a. Oraculo: calibracion POR ESCENARIO (usa el propio escenario de test) ===")
    oracle = per_scenario_oracle_calibration(data)
    for scen, stats in sorted(oracle["per_scenario"].items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  scen {scen}: n={stats['n']:4d} mean_err={stats['mean_error']:6.2f}px "
              f"AUC={stats['auc']:.4f}  (LOSO real era {loso['per_scenario'][scen]['auc']:.4f})")
    print(f"  POOLED oraculo por-escenario: AUC={oracle['pooled_auc']:.4f} "
          f"(vs LOSO real {loso['pooled_auc']:.4f})")

    print("\n=== 3b. Extension del modelo (features gnomonicas de orden superior, LOSO real) ===")
    ext = per_scenario_oracle_tilt_bucket(data, loso)
    for scen, stats in sorted(ext["per_scenario"].items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  scen {scen}: n={stats['n']:4d} mean_err={stats['mean_error']:6.2f}px "
              f"AUC={stats['auc']:.4f}  (modelo base LOSO {loso['per_scenario'][scen]['auc']:.4f})")
    print(f"  POOLED features de orden superior (LOSO real): AUC={ext['pooled_auc']:.4f} "
          f"(vs modelo base LOSO {loso['pooled_auc']:.4f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = dict(
        n_cases=n,
        loso=loso,
        hypothesis_tilt_range=h_tilt,
        hypothesis_gt_scale=h_scale,
        hypothesis_theta_residual=h_resid,
        hypothesis_hidden_scenario_param=h_hidden,
        boresight_mismatch_vs_bias=h_mech,
        coefficient_transfer_ablations=ablations,
        eye_rotation_scenario_correlation=h_eye,
        oracle_per_scenario_calibration=oracle,
        extended_model_loso=ext,
    )
    out_path = OUTPUT_DIR / "scenario_heterogeneity_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
