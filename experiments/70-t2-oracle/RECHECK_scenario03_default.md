# Task 2 oracle signal measurement

Command root: `data\Task 2`; seed: 0; requested cases: 83; loaded: 83; fully skipped: 0; en-face convention: `identity__keep_u__keep_v`; radii mode: `pixels`.

The native en-face grid is **128 x 512** (slice index x lateral pixel), as produced by `src/fido/data/common.py`; all en-face measurements use that actual grid.

## Geometry self-test

Crosshair maximum error: **0.0000 px** (n=1; required <0.01 px).

## A - Fundus vasculature landmarks versus OCT vessels

Cases with volume segmentation: n=83; cases contributing >=1 in-footprint landmark: n=52. In-footprint landmarks per case: mean=1.0602, median=1.0000.

| radius (px) | observed | uniform null | observed/null |
|---:|---:|---:|---:|
| 0 | 0.0064 | 0.0000 | n/a |
| 2 | 0.0503 | 0.0734 | 0.6856 |
| 5 | 0.2250 | 0.2131 | 1.0556 |
| 10 | 0.4314 | 0.4301 | 1.0030 |
| 20 | 0.6753 | 0.5378 | 1.2557 |

## B - Intensity projection signal

Random-translation null uses 200 positions per case. Values are case means; n is reported per projection. Segmentation missing in 0 loaded case(s).

| projection | n | GT NCC | NCC z-score | GT MI | null MI |
|---|---:|---:|---:|---:|---:|
| mean | 83 | -0.0752 | -0.2155 | 0.1630 | 0.2040 |
| MIP | 83 | -0.0493 | -0.1988 | 0.0907 | 0.1075 |
| min | 83 | 0.0140 | 0.2194 | 0.0049 | 0.0055 |
| slab_0 | 83 | -0.0524 | -0.1776 | 0.0861 | 0.0832 |
| slab_1 | 83 | -0.1948 | -0.4920 | 0.2121 | 0.2048 |
| slab_2 | 83 | -0.0097 | -0.0479 | 0.2771 | 0.3263 |
| slab_3 | 83 | 0.2640 | 0.7491 | 0.2695 | 0.2863 |
| gradient | 83 | -0.0506 | -0.1759 | 0.0797 | 0.0992 |
| vessel_density | 56 | 0.0044 | 0.0506 | 0.0038 | 0.0049 |

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
