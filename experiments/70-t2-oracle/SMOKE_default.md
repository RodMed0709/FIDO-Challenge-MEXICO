# Task 2 oracle signal measurement

Command root: `data\Mock Test\Task 2`; seed: 0; requested cases: 5; loaded: 5; fully skipped: 0; en-face convention: `identity__keep_u__keep_v`; radii mode: `pixels`.

The native en-face grid is **128 x 512** (slice index x lateral pixel), as produced by `src/fido/data/common.py`; all en-face measurements use that actual grid.

## Geometry self-test

Crosshair maximum error: **0.0000 px** (n=1; required <0.01 px).

## A - Fundus vasculature landmarks versus OCT vessels

Cases with volume segmentation: n=5; cases contributing >=1 in-footprint landmark: n=2. In-footprint landmarks per case: mean=0.8000, median=0.0000.

| radius (px) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | n/a |
| 2 | 0.5000 | 0.2500 | 2.0000 |
| 5 | 0.5000 | 0.7500 | 0.6667 |
| 10 | 0.5000 | 1.0000 | 0.5000 |
| 20 | 0.5000 | 1.0000 | 0.5000 |

## B - Intensity projection signal

Random-translation null uses 200 positions per case. Values are case means; n is reported per projection. Segmentation missing in 0 loaded case(s).

| projection | n | GT NCC | NCC z-score | GT MI | null MI |
|---|---:|---:|---:|---:|---:|
| mean | 5 | -0.3618 | -1.1028 | 0.3464 | 0.2590 |
| MIP | 5 | -0.2933 | -1.3684 | 0.1865 | 0.1326 |
| min | 5 | -0.0197 | -0.8285 | 0.0100 | 0.0056 |
| slab_0 | 5 | 0.0226 | -0.0030 | 0.0332 | 0.0411 |
| slab_1 | 5 | -0.1430 | -0.5197 | 0.1618 | 0.1723 |
| slab_2 | 5 | -0.1920 | -0.7808 | 0.4535 | 0.3125 |
| slab_3 | 5 | -0.0666 | -1.0273 | 0.2553 | 0.2156 |
| gradient | 5 | -0.2236 | -1.0778 | 0.1615 | 0.0997 |
| vessel_density | 4 | -0.0350 | -0.8225 | 0.0066 | 0.0065 |

## C - Instrument landmarks versus OCT instrument

The OCT mask is the union of official classes 8 (`Forceps`), 10 (`Endoilluminator`), and 11 (`InstrumentInOCT`); class 12 (`ToolMirrorOCTArtifact`) is excluded. This measures shared instrument presence; it does not require assigning an ambiguous generic class to a particular tool.

Cases with segmentation: n=5; cases with >=1 instrument keypoint inside the OCT footprint: 4/5 (0.8000). Cases contributing overlap rates: n=4.

| radius (px) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | n/a |
| 2 | 0.0000 | 0.0000 | n/a |
| 5 | 0.0000 | 0.0000 | n/a |
| 10 | 0.0000 | 0.0000 | n/a |
| 20 | 0.0000 | 0.0000 | n/a |
