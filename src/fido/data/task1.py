from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fido.data.common import load_grayscale, load_label_map, load_rgb
from fido.data.common import group_kfold_indices

CANNULA_INACTIVE_SENTINEL = 1e6  # T1-R1: 2147484000 real, cualquier umbral < eso sirve
ILM_CLASS = 1
INSTRUMENT_CLASS = 11
# Corregido 2026-08-19: los organizadores arreglaron la resolucion de
# profundidad (4000 um / 512 px = 7.8125 um/px). Antes era 10 (implicitamente
# 5000/512). Ver scoring_keypoints.py oficial y RESULTS.md.
GROUND_TRUTH_DISTANCE_SCALE = (4.0 / 512) * 1000  # = 7.8125


@dataclass(frozen=True)
class Task1CaseSplit:
    """Vista ligera de casos que conserva metadatos y sus indices originales."""

    cases: tuple[dict, ...]
    indices: np.ndarray

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @property
    def scenario(self) -> np.ndarray:
        return np.asarray([case["scenario"] for case in self.cases])

    @property
    def case_id(self) -> np.ndarray:
        return np.asarray([f"{case['scenario']}/{case['frame_id']}" for case in self.cases])


def task1_split(cases: list[dict], n_splits: int = 5, fold: int = 0,
                seed: int = 0) -> tuple[Task1CaseSplit, Task1CaseSplit]:
    """GroupKFold por escenario, con indices reutilizables por ``Subset``."""
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits}), got {fold}")
    groups = [case["scenario"] for case in cases]
    case_ids = [f"{case['scenario']}/{case['frame_id']}" for case in cases]
    id_counts = Counter(case_ids)
    if len(id_counts) != len(case_ids):
        duplicates = sorted(case_id for case_id, count in id_counts.items() if count > 1)
        raise ValueError(f"Duplicate Task 1 case IDs: {duplicates[:5]}")
    if len(set(groups)) < n_splits:
        raise ValueError(
            f"Task 1 split requires at least {n_splits} scenarios; got {len(set(groups))}"
        )
    splits = group_kfold_indices(groups, n_splits=n_splits, seed=seed)
    train_idx, val_idx = next(split for split_number, split in enumerate(splits)
                              if split_number == fold)
    result = (
        Task1CaseSplit(tuple(cases[int(i)] for i in train_idx), np.asarray(train_idx)),
        Task1CaseSplit(tuple(cases[int(i)] for i in val_idx), np.asarray(val_idx)),
    )
    observed = set(result[0].case_id) | set(result[1].case_id)
    if set(result[0].case_id) & set(result[1].case_id) or observed != set(case_ids):
        raise RuntimeError("Task 1 split is not an exact, disjoint partition of case IDs")
    return result


def first_task1_split(cases: list[dict], n_splits: int = 5,
                      seed: int = 0) -> tuple[Task1CaseSplit, Task1CaseSplit]:
    return task1_split(cases, n_splits=n_splits, fold=0, seed=seed)


def _active_or_none(json_path: Path, max_retries: int = 2) -> bool:
    """Reintenta el mismo JSON; nunca convierte un fallo I/O en caso inactivo."""
    last_error = None
    for _ in range(max_retries + 1):
        try:
            return _cannula_is_active(json_path)
        except OSError as exc:
            last_error = exc
    case_id = f"{json_path.parent.parent.name}/{json_path.stem}"
    raise OSError(
        f"Failed to inspect Task 1 case {case_id} after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _cannula_is_active(json_path: Path) -> bool:
    """True si la CANNULA tiene una 'ILM Distance' real (no el centinela de
    'no aplica'). Lee el JSON completo -- es la operacion cara del escaneo."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cannula_ilm = (
        data.get("Surgical Tool", {}).get("CANNULA", {}).get("Meta", {}).get("ILM Distance")
    )
    return cannula_ilm is not None and cannula_ilm < CANNULA_INACTIVE_SENTINEL


def _list_scenario_dirs(root: Path, max_retries: int = 2) -> list[Path]:
    last_error = None
    for _ in range(max_retries + 1):
        try:
            return sorted(root.glob("Scenario_*"))
        except OSError as exc:
            last_error = exc
    raise OSError(
        f"Failed to list Task 1 root {root} after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _list_scenario_jsons(numerical_dir: Path, max_retries: int = 2) -> list[Path]:
    last_error = None
    for _ in range(max_retries + 1):
        try:
            if not numerical_dir.is_dir():
                return []
            return sorted(numerical_dir.glob("*.json"))
        except OSError as exc:
            last_error = exc
    scenario = numerical_dir.parent.name
    raise OSError(
        f"Failed to list Task 1 scenario {scenario}/Numerical after "
        f"{max_retries + 1} attempts: {last_error}"
    ) from last_error


def find_task1_cases(root: Path, only_cannula_active: bool = True,
                     max_workers: int = 16) -> list[dict]:
    """MooseFS (fs de red del pod) da OSError/PermissionError transitorios --
    un caso individual roto no debe tronar el escaneo completo (mismo patron
    que find_task2_cases en task2.py).

    El filtro `only_cannula_active` obliga a abrir los 61,691 JSON del dataset.
    Secuencialmente sobre MooseFS eso son ~15 min EN CADA LANZAMIENTO (medido:
    ~15 ms por archivo). Se paraleliza con un pool de hilos porque el cuello es
    I/O de red, no CPU -- el GIL no estorba aqui. Usa `Task1Dataset(cache_path=...)`
    para saltarselo del todo entre corridas.
    """
    cases = []
    for scenario_dir in _list_scenario_dirs(root):
        numerical_dir = scenario_dir / "Numerical"
        json_paths = _list_scenario_jsons(numerical_dir)
        if not json_paths:
            continue

        if only_cannula_active:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                flags = list(pool.map(_active_or_none, json_paths))
        else:
            flags = [True] * len(json_paths)

        for json_path, active in zip(json_paths, flags):
            if only_cannula_active and not active:
                continue
            cases.append({
                "scenario": scenario_dir.name,
                "frame_id": json_path.stem,
                "scenario_dir": scenario_dir,
                "json_path": json_path,
            })
    return cases


def find_task1_cases_cached(root: Path, only_cannula_active: bool = True,
                            cache_path: Path | None = None) -> list[dict]:
    """`find_task1_cases` con cache en disco de la lista de casos.

    El escaneo abre los 61,691 JSON del dataset y su resultado depende solo del
    contenido del dataset, que es estatico -- no hay razon para repetirlo en cada
    lanzamiento. Solo se serializan `scenario` y `frame_id`; los `Path` se
    reconstruyen al cargar, para que el cache no quede atado a rutas absolutas de
    una maquina concreta.
    """
    if cache_path is not None and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("only_cannula_active") != only_cannula_active:
                warnings.warn(
                    f"Cache {cache_path} se genero con only_cannula_active="
                    f"{payload.get('only_cannula_active')}, se pidio {only_cannula_active}. "
                    "Reescaneando.",
                    RuntimeWarning,
                )
            else:
                return [
                    {
                        "scenario": entry["scenario"],
                        "frame_id": entry["frame_id"],
                        "scenario_dir": root / entry["scenario"],
                        "json_path": root / entry["scenario"] / "Numerical" / f"{entry['frame_id']}.json",
                    }
                    for entry in payload["cases"]
                ]
        except (OSError, ValueError, KeyError) as exc:
            warnings.warn(f"Cache {cache_path} ilegible ({exc}). Reescaneando.", RuntimeWarning)

    cases = find_task1_cases(root, only_cannula_active)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({
                    "only_cannula_active": only_cannula_active,
                    "cases": [{"scenario": c["scenario"], "frame_id": c["frame_id"]} for c in cases],
                }),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.warn(f"No se pudo escribir el cache {cache_path}: {exc}", RuntimeWarning)

    return cases


def remap_bscan_labels(label_map: np.ndarray, ilm_class: int = ILM_CLASS,
                        instrument_class: int = INSTRUMENT_CLASS) -> np.ndarray:
    """Remapea el enum GenericLabels crudo a 3 clases para el segmentador
    liviano de T1-R5: 0=fondo, 1=Ilm, 2=InstrumentInOCT (todo lo demás -> 0)."""
    remapped = np.zeros_like(label_map, dtype=np.int64)
    remapped[label_map == ilm_class] = 1
    remapped[label_map == instrument_class] = 2
    return remapped


class Task1Dataset(torch.utils.data.Dataset):
    """`load_fundus` / `load_bscan` NO son cosmeticos -- son la diferencia entre
    leer 1055KB o 356KB por caso sobre un filesystem de red lento:

    | archivo                | tamano | lo usa train_task1_unet | lo usa train_task1_keypoint |
    |------------------------|--------|-------------------------|-----------------------------|
    | microscope.png (fundus)| 699 KB | NO                      | si                          |
    | Bscan/0{0,1}.png       | 352 KB | si                      | NO                          |
    | Bscan/Segmentation/*   |   4 KB | si                      | NO                          |

    Medido en el pod (2026-08-17): el fundus solo es el 38% del tiempo de I/O de
    Task 1, y ademas produce un tensor float32 de 12MB por caso que viaja por IPC
    del worker al proceso principal (100MB por batch de 8) para ser descartado.
    Cada script de entrenamiento debe apagar explicitamente lo que no consume.
    """

    def __init__(self, root: Path, only_cannula_active: bool = True, bscan_size: int = 512,
                 load_fundus: bool = True, load_bscan: bool = True,
                 cache_path: Path | None = None, max_load_retries: int = 2,
                 retry_delay: float = 0.05):
        super().__init__()
        self.root = root
        self.cases = find_task1_cases_cached(root, only_cannula_active, cache_path)
        self.bscan_size = bscan_size
        self.load_fundus = load_fundus
        self.load_bscan = load_bscan
        self.max_load_retries = max_load_retries
        self.retry_delay = retry_delay

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        last_error: OSError | None = None
        for attempt in range(self.max_load_retries + 1):
            try:
                return self._load_case(case)
            except OSError as exc:
                last_error = exc
                retry_delay = getattr(self, "retry_delay", 0.0)
                if attempt < self.max_load_retries and retry_delay:
                    time.sleep(retry_delay * (2 ** attempt))
        case_id = f"{case['scenario']}/{case['frame_id']}"
        raise OSError(
            f"Failed to load Task 1 case {case_id} after "
            f"{self.max_load_retries + 1} attempts: {last_error}"
        ) from last_error

    def _load_case(self, case: dict) -> dict:
        scenario_dir = case["scenario_dir"]
        frame_id = case["frame_id"]

        data = json.loads(case["json_path"].read_text(encoding="utf-8"))
        gt = data["Ground Truth"]["Task 1"]
        keypoint = torch.tensor(gt[:2], dtype=torch.float32)
        distance = torch.tensor(gt[2] / GROUND_TRUTH_DISTANCE_SCALE, dtype=torch.float32)

        bscan_dir = scenario_dir / "iOCT Microscope" / "Bscan" / frame_id
        size = self.bscan_size
        has_oct = bscan_dir.is_dir()

        if not self.load_bscan:
            # `has_oct` se sigue calculando (es un `.is_dir()`, barato) porque el
            # 10% de casos de test sin OCT es parte del contrato del challenge y
            # quien no cargue B-scans igual puede querer saberlo.
            bscan_tensor = None
            seg_tensor = None
        elif has_oct:
            bscans, segs = [], []
            for slice_name in ("00.png", "01.png"):
                slice_path = bscan_dir / slice_name
                if slice_path.exists():
                    bscans.append(load_grayscale(slice_path).astype(np.float32) / 255.0)
                    seg_path = bscan_dir / "Segmentation" / slice_name
                    segs.append(remap_bscan_labels(load_label_map(seg_path)) if seg_path.exists()
                                else np.full((size, size), -1, dtype=np.int64))
                else:
                    bscans.append(np.zeros((size, size), dtype=np.float32))
                    segs.append(np.full((size, size), -1, dtype=np.int64))
            bscan_tensor = torch.from_numpy(np.stack(bscans, axis=0))
            seg_tensor = torch.from_numpy(np.stack(segs, axis=0)).long()
        else:
            bscan_tensor = torch.zeros(2, size, size, dtype=torch.float32)
            seg_tensor = torch.full((2, size, size), -1, dtype=torch.long)

        result = {
            "has_oct": torch.tensor(has_oct, dtype=torch.bool),
            "keypoint": keypoint,
            "distance": distance,
            "scenario": case["scenario"],
            "frame_id": frame_id,
        }

        # Las claves apagadas se OMITEN en vez de devolver ceros: quien las lea
        # sin haberlas pedido revienta con un KeyError claro, en vez de entrenar
        # en silencio contra un tensor de ceros.
        if self.load_fundus:
            fundus = load_rgb(scenario_dir / "Stereo Left" / frame_id / "microscope.png")
            result["fundus"] = (
                torch.from_numpy(fundus.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            )
        if self.load_bscan:
            result["bscan"] = bscan_tensor
            result["bscan_seg"] = seg_tensor

        return result
