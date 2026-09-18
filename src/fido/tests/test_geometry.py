"""Sanity checks de src/fido/geometry.py contra el scoring oficial vendorizado.

No es un test de framework — es la verificación de que compose_similarity +
project_corners + corner_auc reproducen exactamente lo que hace
vendor/fido/Codabench Bundle/scoring_program/scoring_registration.py, antes de
construir cualquier modelo encima. Correr con:

    python -m src.fido.tests.test_geometry
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fido.geometry import (  # noqa: E402
    compose_similarity, decompose_similarity, project_corners, corner_error,
    corner_auc, fit_closed_form_similarity,
)


def load_official_scorer():
    """Importa el módulo de scoring vendorizado tal cual, sin copiarlo."""
    import importlib.util

    path = (PROJECT_ROOT / "vendor" / "fido" / "Codabench Bundle" /
            "scoring_program" / "scoring_registration.py")
    spec = importlib.util.spec_from_file_location("official_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decompose_recompose_within_known_residual():
    """decompose_similarity + compose_similarity sobre una matriz GT real NO
    debe reproducirla byte a byte — T2-R3 (ATTACK_LADDER.md) ya midió que la
    matriz real no es una similitud reflejada pura: residuo medio de 1.74px
    al forzar la forma de 4 DOF, techo de AUC 0.794. Este test verifica que el
    error de esquinas se queda dentro de ese orden de magnitud (no que sea
    cero), usando corner_error/corner_auc — las mismas funciones que puntúan
    de verdad — en vez de comparar la matriz entrada por entrada."""
    candidates = [
        PROJECT_ROOT / "data" / "_annotations" / "Task 2",
        PROJECT_ROOT / "data" / "Mock Test" / "Task 2",
    ]
    sample_paths = [p for root in candidates for p in root.glob("Scenario_*/Numerical/*.json")]
    if not sample_paths:
        raise SystemExit("No hay ningún GT real de Task 2 disponible para este test")

    errors = []
    for sample_path in sample_paths:
        data = json.loads(sample_path.read_text(encoding="utf-8"))
        gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float64)
        params = decompose_similarity(gt_matrix)
        recomposed = compose_similarity(**params, reflect=True)
        errors.append(float(corner_error(recomposed, gt_matrix)))

    errors = np.asarray(errors)
    mean_error = errors.mean()
    # Umbral generoso (5x el residuo medio conocido de 1.74px) para no ser
    # frágil ante la varianza de una muestra chica (Mock Test = 5 casos).
    assert mean_error < 10.0, (
        f"decompose+compose se aleja demasiado del GT real: error medio de "
        f"esquinas {mean_error:.2f}px sobre {len(errors)} casos (se esperaba "
        f"algo del orden de 1.74px, el residuo ya medido en T2-R3)"
    )
    print(f"  OK: decompose+compose da error de esquinas medio {mean_error:.2f}px "
          f"sobre {len(errors)} casos reales (consistente con el residuo de T2-R3)")


def test_decompose_round_trips_random_matrices():
    """decompose_similarity(compose_similarity(p)) == p para params aleatorios,
    con y sin reflexión — antes de usarla en el dataset loader de Task 2."""
    rng = np.random.default_rng(2)
    for reflect in (True, False):
        for _ in range(20):
            theta = rng.uniform(0, 2 * math.pi)
            true_params = dict(
                tx=rng.normal(scale=300), ty=rng.normal(scale=300),
                cos_theta=math.cos(theta), sin_theta=math.sin(theta),
                scale=rng.uniform(50, 300),
            )
            matrix = compose_similarity(**true_params, reflect=reflect)
            recovered = decompose_similarity(matrix, reflect=reflect)
            for key in true_params:
                assert abs(recovered[key] - true_params[key]) < 1e-8, (
                    f"reflect={reflect} {key}: esperado {true_params[key]}, "
                    f"recuperado {recovered[key]}"
                )
    print("  OK: decompose_similarity round-trips 40 matrices aleatorias (reflect=True/False)")


def test_corner_error_matches_official_scorer():
    """corner_error() debe dar el mismo número que corner_error() del scoring
    oficial vendorizado, para matrices arbitrarias (no solo el GT)."""
    official = load_official_scorer()

    rng = np.random.default_rng(0)
    for _ in range(20):
        pred = np.eye(3)
        pred[:2, :2] = rng.normal(scale=100, size=(2, 2))
        pred[:2, 2] = rng.normal(scale=500, size=2)
        ref = np.eye(3)
        ref[:2, :2] = rng.normal(scale=100, size=(2, 2))
        ref[:2, 2] = rng.normal(scale=500, size=2)

        official_pred_corners = official.project_corners(pred)
        official_ref_corners = official.project_corners(ref)
        official_error = official.corner_error(official_pred_corners, official_ref_corners)

        our_error = float(corner_error(pred, ref))
        assert abs(official_error - our_error) < 1e-9, (
            f"discrepancia: oficial={official_error} nuestro={our_error}"
        )
    print("  OK: corner_error coincide con el scoring oficial en 20 casos aleatorios")


def test_auc_matches_official_scorer():
    """corner_auc() debe coincidir con auc_from_errors() del bundle oficial."""
    official = load_official_scorer()
    rng = np.random.default_rng(1)
    errors = rng.uniform(0, 30, size=137)

    official_auc = official.auc_from_errors(list(errors), official.MAX_THRESHOLD_PX)
    our_auc = corner_auc(errors)
    assert abs(official_auc - our_auc) < 1e-12, f"oficial={official_auc} nuestro={our_auc}"
    print(f"  OK: corner_auc coincide con auc_from_errors oficial ({our_auc:.6f})")


def test_closed_form_recovers_known_transform():
    """fit_closed_form_similarity debe recuperar exactamente los parámetros
    usados para generar las esquinas, sin ruido."""
    true_params = dict(tx=123.4, ty=-56.7, cos_theta=math.cos(0.7),
                       sin_theta=math.sin(0.7), scale=180.0)
    matrix = compose_similarity(**true_params, reflect=False)
    corners = project_corners(matrix)

    recovered = fit_closed_form_similarity(corners)
    for key in true_params:
        assert abs(recovered[key] - true_params[key]) < 1e-6, (
            f"{key}: esperado {true_params[key]}, recuperado {recovered[key]}"
        )
    print("  OK: fit_closed_form_similarity recupera parámetros exactos sin ruido")


if __name__ == "__main__":
    tests = [
        test_decompose_recompose_within_known_residual,
        test_decompose_round_trips_random_matrices,
        test_corner_error_matches_official_scorer,
        test_auc_matches_official_scorer,
        test_closed_form_recovers_known_transform,
    ]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - queremos ver TODOS los fallos, no solo el primero
            failures.append((test.__name__, exc))
            print(f"  FALLÓ: {test.__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)}/{len(tests)} checks de geometry.py fallaron.")
        raise SystemExit(1)
    print(f"\nTodos los {len(tests)} checks de geometry.py pasaron.")
