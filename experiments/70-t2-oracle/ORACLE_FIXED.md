# Task 2 oracle signal measurement

Command root: `/workspace/data/Task2`; seed: 0; requested cases: 120; loaded: 120; fully skipped: 0; en-face convention: `transpose__flip_u__flip_v`; radii mode: `fraction`.

The native en-face grid is **128 x 512** (slice index x lateral pixel), as produced by `src/fido/data/common.py`; all en-face measurements use that actual grid.

## Geometry self-test

Crosshair maximum error: **0.0000 px** (n=1; required <0.01 px).

## A - Fundus vasculature landmarks versus OCT vessels

Cases with volume segmentation: n=120; cases contributing >=1 in-footprint landmark: n=82. In-footprint landmarks per case: mean=1.2500, median=1.0000.

| radius (fraction) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0.005 | 0.0915 | 0.1077 | 0.8491 |
| 0.01 | 0.1362 | 0.1606 | 0.8481 |
| 0.02 | 0.2581 | 0.2154 | 1.1981 |
| 0.04 | 0.4593 | 0.4837 | 0.9496 |

## B - Intensity projection signal

Random-translation null uses 200 positions per case. Values are case means; n is reported per projection. Segmentation missing in 0 loaded case(s).

| projection | n | GT NCC | NCC z-score | GT MI | null MI |
|---|---:|---:|---:|---:|---:|
| mean | 120 | 0.0320 | 0.0625 | 0.1522 | 0.1882 |
| MIP | 120 | 0.0286 | 0.1730 | 0.0810 | 0.0981 |
| min | 120 | 0.0082 | -0.0307 | 0.0043 | 0.0052 |
| slab_0 | 120 | -0.0034 | 0.0382 | 0.0654 | 0.0748 |
| slab_1 | 120 | -0.0094 | 0.0443 | 0.1885 | 0.2216 |
| slab_2 | 120 | 0.0450 | 0.1465 | 0.2524 | 0.3057 |
| slab_3 | 120 | 0.0020 | -0.0637 | 0.2334 | 0.2762 |
| gradient | 120 | 0.0140 | 0.1599 | 0.0582 | 0.0700 |
| vessel_density | 99 | 0.0037 | 0.1816 | 0.0038 | 0.0049 |

## C - Instrument landmarks versus OCT instrument

The OCT mask is the union of official classes 8 (`Forceps`), 10 (`Endoilluminator`), and 11 (`InstrumentInOCT`); class 12 (`ToolMirrorOCTArtifact`) is excluded. This measures shared instrument presence; it does not require assigning an ambiguous generic class to a particular tool.

Cases with segmentation: n=120; cases with >=1 instrument keypoint inside the OCT footprint: 66/120 (0.5500). Cases contributing overlap rates: n=66.

| radius (fraction) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0.005 | 0.0000 | 0.0152 | 0.0000 |
| 0.01 | 0.0076 | 0.0152 | 0.5000 |
| 0.02 | 0.0126 | 0.0278 | 0.4545 |
| 0.04 | 0.0455 | 0.0278 | 1.6364 |
