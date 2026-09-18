"""Extract a subset of microscope.png + Numerical/*.json from Task 2 zips
into experiments/97-t2-crosshair-recheck/cache/<Scenario>/... without
unzipping the full archive. Local-only, read-only against the zips."""
from __future__ import annotations
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache"

TARGETS = {
    "Scenario_01": 30,
    "Scenario_05": 30,
    "Scenario_08": 30,
    "Scenario_10": 30,
}


def extract_scenario(scenario: str, n: int) -> int:
    zip_path = REPO_ROOT / "data" / "Task 2" / f"{scenario}.zip"
    out_dir = CACHE / scenario
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        stereo = sorted(
            n2 for n2 in names
            if n2.startswith("Stereo Left") and n2.endswith("microscope.png")
        )
        frame_ids = [Path(p).parent.name for p in stereo]
        step = max(1, len(frame_ids) // n)
        picked = frame_ids[::step][:n]
        count = 0
        for fid in picked:
            json_name = f"Numerical/{fid}.json"
            img_name = f"Stereo Left/{fid}/microscope.png"
            if json_name not in names or img_name not in names:
                continue
            json_out = out_dir / "Numerical" / f"{fid}.json"
            img_out = out_dir / "Stereo Left" / fid / "microscope.png"
            json_out.parent.mkdir(parents=True, exist_ok=True)
            img_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_bytes(z.read(json_name))
            img_out.write_bytes(z.read(img_name))
            count += 1
        return count


if __name__ == "__main__":
    for scenario, n in TARGETS.items():
        count = extract_scenario(scenario, n)
        print(f"{scenario}: extracted {count} cases")
