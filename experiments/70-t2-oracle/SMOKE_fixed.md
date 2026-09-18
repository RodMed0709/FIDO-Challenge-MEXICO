# Task 2 oracle signal measurement

Command root: `data\Mock Test\Task 2`; seed: 0; requested cases: 5; loaded: 5; fully skipped: 0; en-face convention: `transpose__flip_u__flip_v`; radii mode: `fraction`.

The native en-face grid is **128 x 512** (slice index x lateral pixel), as produced by `src/fido/data/common.py`; all en-face measurements use that actual grid.

## Geometry self-test

Crosshair maximum error: **0.0000 px** (n=1; required <0.01 px).

## A - Fundus vasculature landmarks versus OCT vessels

Cases with volume segmentation: n=5; cases contributing >=1 in-footprint landmark: n=2. In-footprint landmarks per case: mean=0.8000, median=0.0000.

| radius (fraction) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0.005 | 0.0000 | 0.0000 | n/a |
| 0.01 | 0.0000 | 0.2500 | 0.0000 |
| 0.02 | 0.0000 | 0.2500 | 0.0000 |
| 0.04 | 0.2500 | 0.7500 | 0.3333 |

## B - Intensity projection signal

Random-translation null uses 200 positions per case. Values are case means; n is reported per projection. Segmentation missing in 0 loaded case(s).

| projection | n | GT NCC | NCC z-score | GT MI | null MI |
|---|---:|---:|---:|---:|---:|
| mean | 5 | 0.2297 | 0.5811 | 0.2297 | 0.2551 |
| MIP | 5 | 0.1100 | 0.3548 | 0.1278 | 0.1289 |
| min | 5 | -0.0187 | -1.0204 | 0.0059 | 0.0052 |
| slab_0 | 5 | 0.0048 | -0.4919 | 0.0594 | 0.0398 |
| slab_1 | 5 | 0.2417 | 1.1223 | 0.2182 | 0.1734 |
| slab_2 | 5 | 0.1661 | 0.4380 | 0.3139 | 0.3053 |
| slab_3 | 5 | -0.2410 | -1.6652 | 0.2727 | 0.2114 |
| gradient | 5 | 0.1241 | 0.5049 | 0.0901 | 0.0966 |
| vessel_density | 4 | -0.0316 | -0.5759 | 0.0108 | 0.0064 |

## C - Instrument landmarks versus OCT instrument

The OCT mask is the union of official classes 8 (`Forceps`), 10 (`Endoilluminator`), and 11 (`InstrumentInOCT`); class 12 (`ToolMirrorOCTArtifact`) is excluded. This measures shared instrument presence; it does not require assigning an ambiguous generic class to a particular tool.

Cases with segmentation: n=5; cases with >=1 instrument keypoint inside the OCT footprint: 4/5 (0.8000). Cases contributing overlap rates: n=4.

| radius (fraction) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0.005 | 0.0000 | 0.0000 | n/a |
| 0.01 | 0.0000 | 0.0000 | n/a |
| 0.02 | 0.0000 | 0.0000 | n/a |
| 0.04 | 0.0000 | 0.0000 | n/a |
