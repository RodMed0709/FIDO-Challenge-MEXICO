# Task 2 oracle signal measurement

Command root: `/workspace/data/Task2`; seed: 0; requested cases: 1214; loaded: 1214; fully skipped: 0.

The native en-face grid is **128 x 512** (slice index x lateral pixel), as produced by `src/fido/data/common.py`; all en-face measurements use that actual grid.

## Geometry self-test

Crosshair maximum error: **0.0000 px** (n=1; required <0.01 px).

## A - Fundus vasculature landmarks versus OCT vessels

Cases with volume segmentation: n=1214; cases contributing >=1 in-footprint landmark: n=632. In-footprint landmarks per case: mean=0.9423, median=1.0000.

| radius (px) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0 | 0.0436 | 0.0557 | 0.7831 |
| 2 | 0.1477 | 0.1618 | 0.9126 |
| 5 | 0.3083 | 0.2990 | 1.0311 |
| 10 | 0.5280 | 0.5149 | 1.0253 |
| 20 | 0.7495 | 0.7373 | 1.0165 |

## B - Intensity projection signal

Random-translation null uses 200 positions per case. Values are case means; n is reported per projection. Segmentation missing in 0 loaded case(s).

| projection | n | GT NCC | NCC z-score | GT MI | null MI |
|---|---:|---:|---:|---:|---:|
| mean | 1214 | -0.1382 | -0.3389 | 0.1712 | 0.1915 |
| MIP | 1214 | -0.0787 | -0.3563 | 0.0887 | 0.0977 |
| min | 1214 | 0.0248 | 0.5210 | 0.0085 | 0.0096 |
| slab_0 | 1214 | -0.0142 | -0.0402 | 0.0417 | 0.0432 |
| slab_1 | 1214 | -0.1281 | -0.3070 | 0.1297 | 0.1244 |
| slab_2 | 1214 | -0.1527 | -0.3190 | 0.2852 | 0.3131 |
| slab_3 | 1214 | 0.2039 | 0.5233 | 0.2155 | 0.2258 |
| gradient | 1214 | -0.1171 | -0.5427 | 0.0918 | 0.0971 |
| vessel_density | 1078 | -0.0029 | -0.0449 | 0.0062 | 0.0080 |

## C - Instrument landmarks versus OCT instrument

The OCT mask is the union of official classes 8 (`Forceps`), 10 (`Endoilluminator`), and 11 (`InstrumentInOCT`); class 12 (`ToolMirrorOCTArtifact`) is excluded. This measures shared instrument presence; it does not require assigning an ambiguous generic class to a particular tool.

Cases with segmentation: n=1214; cases with >=1 instrument keypoint inside the OCT footprint: 279/1214 (0.2298). Cases contributing overlap rates: n=279.

| radius (px) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0 | 0.0042 | 0.0048 | 0.8750 |
| 2 | 0.0090 | 0.0197 | 0.4545 |
| 5 | 0.0119 | 0.0311 | 0.3846 |
| 10 | 0.0239 | 0.0436 | 0.5479 |
| 20 | 0.0789 | 0.0651 | 1.2110 |
