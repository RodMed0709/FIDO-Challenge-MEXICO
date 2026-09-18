from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from fido.data.common import (
    TASK2_ENFACE_CONVENTION,
    canonicalize_task2_enface,
    canonicalize_task2_matrix,
    enface_projection,
    enface_vessel_density,
    load_label_map,
    load_rgb,
    load_volume_label_maps,
    load_volume_slices,
)
from fido.geometry import decompose_similarity

TARGET_PARAM_ORDER = ("tx", "ty", "cos_theta", "sin_theta", "scale")


def _retry_discovery(operation, context: str, attempts: int = 6):
    """Reintenta el mismo recurso; nunca sustituye ni omite datos por I/O."""
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as exc:
            if attempt == attempts - 1:
                raise OSError(f"Task 2 discovery failed for {context} after {attempts} attempts") from exc
            time.sleep(0.05 * (2 ** attempt))


def find_task2_cases(root: Path) -> list[dict]:
    """MooseFS (fs de red del pod) da OSError/PermissionError transitorios en
    `.is_dir()`/`.glob()` sobre directorios individuales — un caso roto no debe
    tronar ni colgar el escaneo completo. Se salta el caso con un aviso."""
    cases = []
    scenario_dirs = _retry_discovery(lambda: sorted(root.glob("Scenario_*")), str(root))
    for scenario_dir in scenario_dirs:
        numerical_dir = scenario_dir / "Numerical"
        if not _retry_discovery(numerical_dir.is_dir, f"{scenario_dir.name}/Numerical"):
            continue
        json_paths = _retry_discovery(lambda: sorted(numerical_dir.glob("*.json")),
                                      f"{scenario_dir.name}/Numerical/*.json")
        for json_path in json_paths:
            frame_id = json_path.stem
            volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
            has_volume = _retry_discovery(
                lambda: volume_dir.is_dir() and any(volume_dir.glob("*.png")),
                f"{scenario_dir.name}/{frame_id}",
            )
            if has_volume:
                cases.append({
                    "scenario": scenario_dir.name,
                    "frame_id": frame_id,
                    "scenario_dir": scenario_dir,
                    "json_path": json_path,
                })
    return cases


class Task2Dataset(torch.utils.data.Dataset):
    """Costes por caso medidos en el pod (2026-08-17), para entender los flags:

    | dato                       | archivos | tamano   |
    |----------------------------|----------|----------|
    | Volume/*.png (para enface) |      128 | 21.5 MB  |
    | microscope.png (fundus)    |        1 |  653 KB  |
    | arteriesorveins.png        |        1 |   20 KB  |

    El volumen se lee entero solo para promediarlo sobre el eje de profundidad y
    quedarse con una imagen de 128x512. Como esa proyeccion es una funcion pura
    de datos estaticos, `enface_cache_dir` permite leerla precomputada desde un
    `.npz` (65 KB) en vez de recorrer los 128 PNG -- ~330x menos I/O, que es la
    diferencia entre una epoca de horas y una de segundos.
    Ver `infra/precompute_task2_enface.py`.
    """

    def __init__(self, root: Path, include_vessel_enface: bool = True,
                 include_vessel_mask: bool = True,
                 enface_cache_dir: Path | None = None):
        super().__init__()
        self.root = root
        self.cases = find_task2_cases(root)
        self.include_vessel_enface = include_vessel_enface
        self.include_vessel_mask = include_vessel_mask
        self.enface_cache_dir = enface_cache_dir

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        try:
            return self._load_case(case)
        except OSError as exc:
            case_id = f"{case['scenario']}/{case['frame_id']}"
            raise OSError(f"Failed to load Task 2 case {case_id}; refusing to substitute "
                          "another case because that would contaminate the fold") from exc

    def _load_case(self, case: dict) -> dict:
        scenario_dir = case["scenario_dir"]
        frame_id = case["frame_id"]

        cached = None
        if self.enface_cache_dir is not None:
            cache_file = self.enface_cache_dir / case["scenario"] / f"{frame_id}.npz"
            if cache_file.exists():
                cached = np.load(cache_file)

        if cached is not None:
            fundus = cached["fundus"]
            if "enface_convention" in cached:
                cache_convention = str(cached["enface_convention"].item())
                if cache_convention != "native_slice_lateral_v1":
                    raise ValueError(f"unsupported cached en-face convention: {cache_convention}")
            enface_native = cached["enface"].astype(np.float32) / 255.0
        else:
            fundus = load_rgb(scenario_dir / "Stereo Left" / frame_id / "microscope.png")
            volume = load_volume_slices(scenario_dir / "iOCT Microscope" / "Volume" / frame_id)
            enface_native = enface_projection(volume).astype(np.float32) / 255.0

        enface = canonicalize_task2_enface(enface_native)

        fundus_tensor = torch.from_numpy(fundus.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        enface_tensor = torch.from_numpy(enface).unsqueeze(0)

        data = json.loads(case["json_path"].read_text(encoding="utf-8"))
        native_gt_matrix = np.array(data["Ground Truth"]["Task 2"], dtype=np.float32)
        gt_matrix = canonicalize_task2_matrix(native_gt_matrix)
        gt_matrix_tensor = torch.from_numpy(gt_matrix)

        # C has determinant -1, so reflected native GT becomes a proper
        # similarity in the canonical image frame.
        params = decompose_similarity(gt_matrix, reflect=False)
        target_params = torch.tensor([float(params[k]) for k in TARGET_PARAM_ORDER], dtype=torch.float32)

        result = {
            "fundus": fundus_tensor,
            "enface": enface_tensor,
            "gt_matrix": gt_matrix_tensor,
            "native_gt_matrix": torch.from_numpy(native_gt_matrix),
            "target_params": target_params,
            "scenario": case["scenario"],
            "frame_id": frame_id,
            "enface_convention": TASK2_ENFACE_CONVENTION,
        }

        # Las claves apagadas se OMITEN, no se rellenan con ceros: quien las lea
        # sin pedirlas revienta con un KeyError claro en vez de entrenar contra
        # un tensor vacio en silencio.
        if self.include_vessel_mask:
            vessel_mask_path = scenario_dir / "Stereo Left" / frame_id / "Segmentation" / "arteriesorveins.png"
            vessel_mask = (load_label_map(vessel_mask_path) > 0).astype(np.float32)
            result["fundus_vessel_mask"] = torch.from_numpy(vessel_mask).unsqueeze(0)

        if self.include_vessel_enface:
            volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
            seg_volume = load_volume_label_maps(volume_dir / "Segmentation")
            vessel_enface = canonicalize_task2_enface(enface_vessel_density(seg_volume))
            result["enface_vessel"] = torch.from_numpy(vessel_enface).unsqueeze(0)

        return result


class Task2TransformSubset(torch.utils.data.Dataset):
    """Subset that exposes the original case index to deterministic transforms."""

    def __init__(self, dataset: Task2Dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        source_index = self.indices[index]
        sample = self.dataset[source_index]
        return self.transform(sample, source_index) if self.transform is not None else sample


def load_task2_scales(cases: list[dict]) -> np.ndarray:
    """Read GT scale metadata without loading images or OCT volumes."""
    scales = []
    for case in cases:
        data = json.loads(case["json_path"].read_text(encoding="utf-8"))
        matrix = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)
        scales.append(float(decompose_similarity(matrix)["scale"]))
    return np.asarray(scales, dtype=np.float64)
