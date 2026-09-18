from __future__ import annotations

import time
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold

T = TypeVar("T")


def _retry_on_io_errors(retries: int = 6, delay: float = 0.5) -> Callable:
    """MooseFS (fs de red del pod) da OSError/PermissionError transitorios en
    lecturas -- casi siempre se resuelven al reintentar, pero no siempre en
    menos de 1s (visto en produccion: 3 intentos de 0.3s agotados sin
    resolverse). Backoff exponencial (0.5, 1, 2, 4, 8s) para cubrir cortes
    mas largos sin bloquear indefinidamente. No es un bug de codigo."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except (OSError, PermissionError) as exc:
                    last_exception = exc
                    if attempt < retries - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator


def list_scenario_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Scenario_"))


def load_generic_labels_enum(vendor_root: Optional[Path] = None) -> Optional[object]:
    """Importa dinámicamente `GenericLabels` desde el repo vendorizado (nunca se
    edita ni se copia a mano — misma técnica que test_geometry.py::load_official_scorer).
    Cae a `None` sin lanzar excepción si el vendor no está disponible; quien llama
    debe tener un fallback a los enteros documentados (Ilm=1, InstrumentInOCT=11,
    ArteriesOrVeins=3)."""
    if vendor_root is None:
        vendor_root = Path(__file__).resolve().parents[3] / "vendor" / "fido"
    constants_path = vendor_root / "Dataset Explorer" / "constants.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("fido_constants", constants_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"No se pudo cargar spec desde {constants_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "GenericLabels", None)
    except Exception as exc:
        warnings.warn(
            f"No se pudo importar GenericLabels desde {constants_path}: {exc}. "
            "Usando enteros documentados como fallback.",
            RuntimeWarning,
        )
        return None


@_retry_on_io_errors()
def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


@_retry_on_io_errors()
def load_grayscale(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


@_retry_on_io_errors()
def load_label_map(path: Path) -> np.ndarray:
    """Sin `.convert(...)`: preserva los valores enteros del enum tal cual los
    guardó el simulador (modo "P" o "I", no forzar a "L" que reescala)."""
    return np.array(Image.open(path))


@_retry_on_io_errors()
def _list_pngs(volume_dir: Path) -> list[Path]:
    return sorted(volume_dir.glob("*.png"))


def load_volume_slices(volume_dir: Path) -> np.ndarray:
    slices = _list_pngs(volume_dir)
    return np.stack([load_grayscale(p) for p in slices], axis=0)


def load_volume_label_maps(volume_dir: Path) -> np.ndarray:
    slices = _list_pngs(volume_dir)
    return np.stack([load_label_map(p) for p in slices], axis=0)


def enface_projection(volume: np.ndarray) -> np.ndarray:
    """Misma fórmula exacta que analysis/render_task2_figures.py::enface() —
    fuente de verdad ya usada para las figuras del explicador. volume shape
    (n_slices, profundidad, ancho); promediar sobre el eje 1 (profundidad)."""
    projection = volume.mean(axis=1)
    projection = projection - projection.min()
    projection = projection / max(projection.max(), 1e-6)
    return (projection * 255.0).astype(np.uint8)


TASK2_ENFACE_CONVENTION = "transpose__flip_u__flip_v"
TASK2_CANONICAL_TO_NATIVE = np.array(
    [[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
)


def canonicalize_task2_enface(enface):
    """Map native ``(slice, lateral)`` arrays into Task 2 canonical ``(v,u)``.

    The train-only instrument oracle selects ``transpose__flip_u__flip_v``:
    canonical ``(u,v)`` samples native ``(1-u,1-v)`` after exchanging axes.
    Spatial axes are always the final two, so NumPy arrays and CHW tensors use
    exactly the same single implementation.
    """
    if enface.ndim < 2:
        raise ValueError("enface must have at least two spatial dimensions")
    if isinstance(enface, torch.Tensor):
        return enface.flip((-2, -1)).transpose(-2, -1).contiguous()
    return np.ascontiguousarray(np.swapaxes(np.flip(enface, axis=(-2, -1)), -2, -1))


def canonicalize_task2_matrix(native_matrix):
    """Compose native-grid GT with the canonical-to-native point transform."""
    if isinstance(native_matrix, torch.Tensor):
        convention = native_matrix.new_tensor(TASK2_CANONICAL_TO_NATIVE)
        return native_matrix @ convention
    matrix = np.asarray(native_matrix)
    return matrix @ TASK2_CANONICAL_TO_NATIVE.astype(matrix.dtype, copy=False)


def nativeize_task2_matrix(canonical_matrix):
    """Convert an internal canonical prediction back to the official native GT frame."""
    return canonicalize_task2_matrix(canonical_matrix)  # C is its own inverse.


def enface_vessel_density(seg_volume: np.ndarray, vessel_class: int = 3) -> np.ndarray:
    """Fracción de profundidad ocupada por la clase de vaso en cada columna
    (n_slices, ancho), valores en [0,1]."""
    return (seg_volume == vessel_class).astype(np.float32).mean(axis=1)


def group_kfold_indices(groups: list[str], n_splits: int, seed: int = 0):
    """Wrapper fino sobre sklearn.model_selection.GroupKFold.

    GroupKFold no tiene random_state/shuffle propio (el split es determinista
    dado el orden de los grupos) — `seed` se acepta solo por consistencia de
    firma con otros helpers de este proyecto, no tiene efecto.
    Devuelve el generador de (train_idx, val_idx) de sklearn directamente.
    """
    del seed
    gkf = GroupKFold(n_splits=n_splits)
    return gkf.split(X=np.zeros(len(groups)), groups=groups)
