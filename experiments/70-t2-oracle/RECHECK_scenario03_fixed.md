# Task 2 oracle signal measurement

Command root: `data\Task 2`; seed: 0; requested cases: 83; loaded: 83; fully skipped: 0; en-face convention: `transpose__flip_u__flip_v`; radii mode: `pixels`.

The native en-face grid is **128 x 512** (slice index x lateral pixel), as produced by `src/fido/data/common.py`; all en-face measurements use that actual grid.

## Geometry self-test

Crosshair maximum error: **0.0000 px** (n=1; required <0.01 px).

## A - Fundus vasculature landmarks versus OCT vessels

Cases with volume segmentation: n=83; cases contributing >=1 in-footprint landmark: n=52. In-footprint landmarks per case: mean=1.0602, median=1.0000.

| radius (px) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0 | 0.0375 | 0.0423 | 0.8864 |
| 2 | 0.1381 | 0.1080 | 1.2789 |
| 5 | 0.2112 | 0.2356 | 0.8966 |
| 10 | 0.3692 | 0.4375 | 0.8440 |
| 20 | 0.5298 | 0.6923 | 0.7653 |

## B - Intensity projection signal

Random-translation null uses 200 positions per case. Values are case means; n is reported per projection. Segmentation missing in 0 loaded case(s).

| projection | n | GT NCC | NCC z-score | GT MI | null MI |
|---|---:|---:|---:|---:|---:|
| mean | 83 | 0.0174 | 0.1193 | 0.1667 | 0.2033 |
| MIP | 83 | 0.0239 | 0.1949 | 0.0863 | 0.1066 |
| min | 83 | -0.0030 | -0.1461 | 0.0049 | 0.0055 |
| slab_0 | 83 | -0.0471 | -0.2133 | 0.0792 | 0.0823 |
| slab_1 | 83 | -0.0819 | -0.0673 | 0.1790 | 0.2068 |
| slab_2 | 83 | 0.0956 | 0.3216 | 0.2755 | 0.3294 |
| slab_3 | 83 | 0.0535 | -0.0224 | 0.2347 | 0.2899 |
| gradient | 83 | 0.0483 | 0.3436 | 0.0829 | 0.0999 |
| vessel_density | 56 | 0.0071 | 0.1342 | 0.0038 | 0.0048 |

## C - Instrument landmarks versus OCT instrument

The OCT mask is the union of official classes 8 (`Forceps`), 10 (`Endoilluminator`), and 11 (`InstrumentInOCT`); class 12 (`ToolMirrorOCTArtifact`) is excluded. This measures shared instrument presence; it does not require assigning an ambiguous generic class to a particular tool.

Cases with segmentation: n=83; cases with >=1 instrument keypoint inside the OCT footprint: 0/83 (0.0000). Cases contributing overlap rates: n=0.

| radius (px) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0 | n/a | n/a | n/a |
| 2 | n/a | n/a | n/a |
| 5 | n/a | n/a | n/a |
| 10 | n/a | n/a | n/a |
| 20 | n/a | n/a | n/a |
