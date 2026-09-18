# FIDO-Challenge-MEXICO 🇲🇽

Entry for the **[FIDO Challenge 2026](https://www.codabench.org/competitions/16804)** — *Fusion for
Intelligent Decision-support in Ophthalmology*, MICCAI 2026. Multimodal AI for vitreoretinal
surgery: fuse the surgical microscope view with intraoperative OCT (iOCT) so the system can tell
the surgeon where the instrument tip is, how close it is to the retina, and where the OCT scan
actually landed.

Organized by SynthesEyes GmbH, Carl Zeiss Meditec AG, TU Munich and the Rotterdam Eye Hospital.
Official code: [`SynthesEyes-GmbH/fido`](https://github.com/SynthesEyes-GmbH/fido).

> 📌 **TODO — team description.** Members, affiliations and roles go here.
> _(placeholder: to be completed)_

**We did not win, and this repository is the reason it is still worth reading.** Our best Task 1
submission scored **0.569** against a leader at **0.619**; our Task 2 submission scored **0.00**,
honestly. What we have instead is a fully instrumented record of *why*: a reverse-engineered
submission contract that the official documentation gets wrong in six places, a Task 1 distance
head derived from geometry instead of blind regression, and — found too late to ship — the
measurement showing that **Task 2 is not an image-registration problem at all.**

---

## 📊 Results

### Official leaderboard, Competition phase

Queried from `https://www.codabench.org/api/phases/27554/get_leaderboard/`.

| Task 1 — keypoints + distance | score | | Task 2 — registration | score |
|---|---|---|---|---|
| kwonmj | **0.619** | | cnielsen | **0.475** |
| Alaa Senjab | 0.618 | | Alex | 0.376 |
| akkkkk | 0.604 | | oli4more | 0.280 |
| cnielsen | 0.588 | | kwonmj | 0.279 |
| dancies | 0.565 | | dancies | 0.244 |
| campana | 0.549 | | tianmaxingkong | 0.145 |
| Alex | 0.479 | | akkkkk | 0.111 |
| tianmaxingkong | 0.418 | | 4 teams | 0.000 |

Task 1 column as of 2026-08-14; Task 2 column as of 2026-08-19, after four teams that had been at
zero climbed above it.

**The leader's Task 1 score decomposes as `0.6189 = 0.7 × 0.7973 + 0.3 × 0.2027`.** Ours was
`0.5386 = 0.7 × 0.7309 + 0.3 × 0.0900` — verified exact to the published digit. So the gap was
never in the keypoint. It was in the distance component, where we were collecting **0.027 of the
0.30 on offer**.

### Our submissions

| # | What it was | Local (Mock Test, n=5) | Codabench | Verdict |
|---|---|---|---|---|
| `r00-smoke` | trivial `inference.py` + placeholder weights | T1 0.0000 / T2 0.0000 | not submitted | contract verified end to end |
| `r01-task1-keypoint` | keypoint ep.4 + distance ep.2 | 0.4800 | **0.5386** (`kp=0.7309`, `dist=0.0900`) | scored |
| `r02-best` | keypoint final + distance 0.5709 | 0.5945 | not submitted | local only |
| `r03-dist-v2` | new distance head `unet_dist_v2` | 0.6000 | **0.569** | scored, 6th of 9 |
| `r04-interim-joint` | clean CNN keypoint ep.5 + T2 scale-aug baseline | T1 0.6255 / **T2 0.0000** | **0.561** T1, **0.00** T2 | scored |
| `r05-dist-rescaled` | *bit-identical weights to r04*; only the two distance constants ×1.28 | 0.6382 | 0.569 ⚠️ *(see below)* | contested |
| `r06-fallback-fixed` | r05 + rescaled fallback constant | not recorded | **0.5691** (`kp=0.7573`) | scored |

🔴 **Two honest caveats we are not going to hide.**

1. **`r05` is contested inside our own repository.** `RESULTS.md` records it as submitted and
   scored 0.569; our own adversarial review found that no file in the repo records the upload, and
   called the 0.569 an unsourced premise. Both entries are from the same day. We never resolved it.
2. **We do not appear on the Competition-phase leaderboard.** All twelve entries belong to other
   teams and no `0.569` exists there. Whether our submissions registered in the phase and were not
   published, or never registered at all, is unresolved.

Also unverified: after the organizers corrected the distance scoring constants on 2026-08-19, we
confirmed the change in their GitHub repository but **never confirmed that Codabench redeployed the
corrected scorer.** Every cross-submission comparison spanning that date is therefore on soft
ground, ours included.

---

## 🔑 The submission contract — the part worth stealing

Four of the eleven teams on the Task 2 leaderboard sat at exactly `0.000`. We do not think that is
because four models were bad. **It is because the official documentation contradicts the ingestion
code in ways that produce a silent zero.** We reverse-engineered the contract by reading
`Codabench Bundle/` at commit `5a84d17`, not the web page. The full write-up is in
[`RULES_OF_ENGAGEMENT.md`](RULES_OF_ENGAGEMENT.md); the traps:

| # | The trap | What actually happens |
|---|---|---|
| 1 | **Signature is four arguments**, `inference(task_id, oct_volume, opmi_image, model)` | The docs show three. The ingestion code itself carries the comment *"The documentation may show a different signature; this one is correct."* |
| 2 | **Weights are `model_0.pth` / `model_1.pth`**, never `model.pth` | `REQUIRED_FILES` is validated and a missing name **aborts the whole run**. One character of difference is a zero. |
| 3 | **The zip must be flat** | Zipping the folder instead of its contents is a zero. |
| 4 | **Never declare `numpy` or `torch` in `requirements.txt`** | Already installed; declaring them triggers a reinstall that can break the environment or eat the time budget. |
| 5 | **Shapes differ from the docs and nothing is resized** | `opmi_image` documented 512×512×3, actually **1024×1024×3 uint8**. `oct_volume` Task 1 documented (2,256,256), actually **(2,512,512)**; Task 2 documented (256,256,256), actually **(128,512,512)**. Read `.shape`. |
| 6 | **`oct_volume` can be `None`** | Task 1 forces it on **10% of cases**, chosen with seed `2027`. One unhandled exception **aborts the entire run**, not just that case. |
| 7 | **The real time limit is the global one** | 20 s/case sounds generous, but the Final Round global timeout is **600 s for 100 cases**. Loading 128 PNGs costs 459 ms/case outside the timed window but inside the global one — leaving about **5.5 s/case** of actual budget. |

Nothing of ours ever went to Codabench without passing [`eval/run_local.py`](eval/), which imports
the vendored ingestion and scoring untouched.

---

## ✅ What worked

**1. The tool–tissue distance is measurable, not something to regress blindly.**
`distance_GT = 0.7320 · pixel_gap + 2.6779`, **R² = 0.9916 over 92,012 measurements across 61,691
frames**. The CANNULA is the active instrument in **100%** of frames, and
`GT[2] = CANNULA.Meta['ILM Distance'] × 100` held exactly in 30 of 30 sampled cases. That turns a
regression problem into segmentation plus arithmetic.

**2. Applying the organizers' scale correction by exact algebra was the only net distance gain we
ever measured.** When they changed `GROUND_TRUTH_DISTANCE_SCALE` from 10 to 7.8125 and the
threshold from 10 to 20 px, we rescaled the two fitted constants by 1.28 — **without retraining
anything.** Isolated control on the Mock Test, `keypoint_auc` identical across all three runs so
distance is the only moving part:

| run | scorer | `distance_auc` | Task 1 |
|---|---|---|---|
| `r04` | old | 0.0909 | 0.6255 |
| `r04` | new | **0.0000** | 0.5982 |
| `r05` (rescaled) | new | **0.1333** | **0.6382** |

**Raising the threshold buys nothing on its own** — under the corrected scorer the old prediction
collapses to exact zero, meaning its error exceeded 20 px in every case. The entire gain is the
rescale. Independent confirmation: the fitted slope moves from 0.732 to 0.937, and it should be
≈1.0 — a 27% deviation becomes 6.3%.

**3. A decoder audit removed a ceiling that made Task 2 unscorable.** Fed **perfect** ground-truth
localization and parameters, the old decoder returned 112.88 px error and `AUC 0.0000`; the fixed
one returned 1.81 px and **`AUC 0.7848`**. Three bugs — centre-vs-corner convention, a half-cell
bias, a 4:1 template aspect. After the fix the real model produced `val_corner_auc = 0.0048`, the
first non-zero of the project.

**4. Exact geometric scale augmentation, as the only change.** Synthetic holdout, 496 evaluations:
AUC `0.000550 → 0.006048`, mean error `85.765 → 65.305 px`, **scale MAE `31.163 → 14.278 px`
(−54.2%)** against a pre-registered 20% gate, with no scenario regressing.

**5. Half the training slowness was code reading data it never used.** Precomputed en-face
`812.7 → 20.7 ms/case` (**39×**), Task 1 dataset on local disk `108 → 9.1 ms/case` (**12×**),
scanning 61,691 JSON files `~15 min → 67.5 s → ~0.5 s` cached, GPU from oscillating 0–78% to
sustained 100%. The precomputed en-face was verified **bit-identical** to the original path.

---

## ❌ What did not work — published as results

A faithful negative is cheaper for the next team than repeating it. Each of these closed a rung.

| Rung | Hypothesis | The number that killed it |
|---|---|---|
| T2-R4 | Bridge iOCT→fundus through vessels with classical template matching | With **perfect GT masks**: mean corner error **573.69 px**, `corner_auc = 0.0000`. A self-consistent synthetic case passes at 0.09 px, so the code is right and the signal is not there. |
| T2-R4 rd.2 | Some other vessel representation rescues it | Oracle NCC: density 0.0071, binarized 0.0071, MIP mathematically identical, dilation 2–21 px stays in [0.001, 0.018]. Free translation search reaches 0.24–0.31 but **lands away from GT in 4/4 cases** — vascular self-similarity, a false positive. |
| T2-82 | Learn common CNN descriptors | `train_loss` falls five orders of magnitude (1.8e-2 → 3.7e-7): trivial collapse. Decisive control: **`shuffle_drop_points` oscillates between −0.567 and +0.151** — shuffling the OCT across cases degrades nothing, so the model never used the OCT. |
| T2-83 / T2-R11 | Vasculature correlates in *some* orientation | 906 points, **32 observed/null ratios in [0.81, 1.04]** across all eight `{identity,transpose}×{flip_u}×{flip_v}` conventions. Only the *instrument* passed the gate (ratios 3.0–3.6). |
| T1/T2 | A stronger backbone is the bottleneck | Same split, n=14,399: CNN from scratch **0.8539**, ResNet-18+FPN pretrained **0.8534**, DINOv2 **0.8005**. The first two differ by **0.0005**. The ResNet's `heatmap_bce` is an order of magnitude lower and it still does not win AUC. |
| T1 | TTA and ensembling lift the keypoint | Baseline `auc 0.8697 / 0.91 px`. **TTA with flips+rotations: `auc 0.0061 / 70.62 px`.** Not a de-transform bug — 45 synthetic tests verify every transform and its inverse to ~0 px. It is anatomy: the retina has a preferred orientation (optic disc on one side), and a flip produces an anatomically impossible image. |
| T2-101 | The honest end-to-end number | Replacing GT rotation with rotation predicted from the OCT volume alone (nested LOSO, no leakage): **AUC 0.0004**, mean error 122.82 px. Pre-registered gate was `< 0.05` → **no submission was built.** |
| T2-R2 | Detect the iOCT crosshair in the microscope image | **It is not rendered there.** The shortcut does not exist. |
| — | Some constant prediction gets partial credit on Task 2 | A constant predictor scores **exactly 0.000**. So does a constant-pose baseline. There is no free floor, unlike Task 1 where a constant distance is worth 0.0445. |
| — | Train and test share a scale regime | They do not overlap at all: training scale lies in **[126.66, 188.09]**, the Mock Test in **[213.81, 226.16]**. Any model that does not generalize across scale is capped near zero before it starts. |

---

## 🔬 Task 2: the reframing that came too late

Five independent internal oracles concluded Task 2 had no exploitable signal. Then the leaderboard
showed seven teams above zero and a leader at **0.475**. Our own record says it plainly:

> 🔴 *"The leaderboard refutes our Task 2 verdict. The five internal closures for 'absence of
> signal' are incompatible with this empirical fact. Somewhere in our chain there is an error:
> either the oracles measure what does not matter, or an upstream convention/geometry failure
> contaminates all of them. Task 2 is reopened."*

For context on how much was on the table: feeding the scorer the **perfect ground-truth matrix**
yields AUC **0.6389** — so the leader at 0.475 was extracting a large fraction of what the
parameterization allows. This was never a problem with no signal in it.

We ran a dedicated audit (T2-94) looking for the upstream defect: the en-face pipeline, the axis
convention, the GT matrix against the scorer, oracle self-consistency. **It found no code,
convention or geometry bug** — which is itself a useful negative, because it forced the search
somewhere else. It also resolved a documentary contradiction: what looks to the eye like aligned
vasculature between the two modalities is a real *perception* but not pixel-precision vasculature —
a gestalt illusion reinforced by instrument shadows and low-frequency artifacts.

Two causes, both confirmed:

**(a) An upstream convention bug.** The en-face projection was being built with
`identity__keep_u__keep_v`. Sweeping all eight conventions over 1,214 cases, the correct one is
**`transpose__flip_u__flip_v`** — observed/null ratios 3.64/3.43/4.18/3.16 against 1.17/1.16/0.75/0.70
for the one in use. This partially invalidates T2-R11: its instrument negatives could not close
anything. Honest qualifier: the fix rescues the *instrument* signal, not the vessel signal.

**(b) The larger error — we were solving the wrong problem.** Task 2 is **not appearance-based
image registration. It is 3D pose estimation.** Across 1,214 cases the annotations show only two
things varying: `iOCT.Rotation` and `Eyeball.Rotation`. The microscope is fixed and nothing
translates. The ground-truth matrix derives from those with **R² 0.975–0.994**, and in fact depends
on `iOCT.Rotation` alone (`theta = -roll + theta0`, residual 0.78°).

From there the chain is short: scenario heterogeneity isolates to eight projection coefficients
(recalibrating them per scenario gives AUC **0.7204**), and those coefficients can be estimated from
the fundus using a single feature — the radius of the optical vignetting — for AUC **0.3201**.

⚠️ **Do not cite 0.1512 / 0.3201 / 0.7204 as expected scores.** They are ceilings *conditioned on
ground-truth rotation*, and our own constitution forbids quoting them otherwise. The only
deployable number we ever produced is **0.0004**, because predicted rotation carries 14.30° median
error and the geometric model collapses to zero at 2–5°. **Task 2 was characterized, not solved.**

The residual problem, stated cleanly for whoever wants it: *estimate `iOCT.Rotation` from the OCT
volume to better than 5°.* The geometry downstream is already verified and delivers 0.32–0.72.

---

## 🧪 How we measure

- **One hypothesis per rung.** If a change brings three new things and the score moves, nothing was
  learned. Rungs carry an explicit state and live in [`ATTACK_LADDER.md`](ATTACK_LADDER.md).
- **Pre-register the expectation and the numeric gate before running**, in
  `experiments/*/PRE_REGISTRATION.md`. This is what separates "it worked" from "something happened
  and I rationalized it afterwards."
- **A rung with no written result counts as not executed**, and no number enters
  [`RESULTS.md`](RESULTS.md) without the command and the commit that produced it.
- **GroupKFold by scenario, always.** Frames from one scenario share near-identical anatomy;
  splitting by frame leaks.
- **The Mock Test is 5 cases.** It verifies that the container does not crash. It does **not**
  select models — and it demonstrably does not predict Codabench: `r01` went 0.4800 local → 0.5386
  real, `r03` went 0.6000 local → 0.569 real. Optimistic once, pessimistic once.
- **`vendor/fido/` is frozen and never edited.** It is what makes local scoring trustworthy.

---

## 🗂️ Data, and credit where it is due

The challenge data — 1,214 Task 2 snapshots and 61,691 Task 1 frames, about 83 GB compressed — is
the organizers', released under **CC BY-NC-ND**: non-commercial, no redistribution. **None of it is
in this repository**: no frames, no volumes, no annotations, and no renders or figures derived from
them. Several of our own explainer pages exist only as image-free templates here for exactly this
reason. Request the data from the organizers.

Two numbers worth knowing before you plan compute: **the dataset is roughly 8× larger than
announced** (documentation said ~150 Task 2 snapshots and ~25,000 Task 1 frames), and extracted it
occupies about **495 GB**.

[`vendor/fido/`](vendor/) is a verbatim copy of the organizers' public repository at commit
`5a84d17` (2026-08-08), kept with its licence intact, because trustworthy local scoring requires an
unmodified copy. `reference_data/` is a placeholder — no challenge data travels with it.

Three of the seven organizers published **arXiv 2603.25555**, which solves Task 1 on what is almost
certainly the same simulator (kp_dist 7.93 ± 0.47 px, dMAE 128.32 ± 20.03 µm). It is the reference
any Task 1 result should be read against.

## ⚖️ Licence

Code, notebooks, configs and docs in this repository: [Apache-2.0](LICENSE).

No model weights are published here. They are a research artifact from a benchmark challenge and in
any case **not a medical device — nothing here may be used for clinical decision-making.**

## 🧭 How to read this repository

| Path | What |
|---|---|
| [`RULES_OF_ENGAGEMENT.md`](RULES_OF_ENGAGEMENT.md) | **start here if you are competing** — the real submission contract, which is not the documented one |
| [`ATTACK_LADDER.md`](ATTACK_LADDER.md) | every rung, its pre-registered expectation and its actual outcome — including the failures |
| [`RESULTS.md`](RESULTS.md) | one line per run with a number, the command and the commit that produced it |
| [`CONSTITUTION.md`](CONSTITUTION.md) | the non-negotiable rules |
| [`THE_MAP.md`](THE_MAP.md) | what exists and in what state |
| [`experiments/<id>/`](experiments/) | one folder per experiment: pre-registration, logs, results |
| [`submissions/<id>/`](submissions/) | the shipped `inference.py` and what it was verified against — weights not included |
| [`ledger/`](ledger/) | structured record: runs with seed, GPU and known defects; facts with an expiry date; decisions; defects |
| [`POD_RESCUE.md`](POD_RESCUE.md) | how the training infrastructure was decommissioned and what was preserved |

## Setup

```bash
git clone https://github.com/RodMed0709/FIDO-Challenge-MEXICO.git
cd FIDO-Challenge-MEXICO

python -m venv .venv && source .venv/bin/activate

# torch first, from the CUDA 12.8 index — see the warning below
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`infra/bootstrap_pod.sh` does the same thing on a fresh RunPod instance and then verifies the GPU
before anything else runs.

Training ran on an **RTX 5090** (Blackwell, `sm_120`, 32 GB) at $0.99/h. That architecture
**requires PyTorch ≥ 2.7 with CUDA 12.8** — cu121 and cu124 wheels import cleanly and then fail on
the first CUDA operation, which costs an afternoon if you do not know it in advance. The bootstrap
script asserts `sm_120` is in `torch.cuda.get_arch_list()` and runs a real matmul before handing
the machine over.

Total recorded compute spend across the project is roughly **$13**, of which **$4.40 was burned on
idle pods** — one created before there was anything to run, one left on for two hours after its
runs finished. A forgotten pod costs about $24/day.

## Team

> 📌 **TODO — team members.** Names, affiliations, ORCIDs and roles.
> _(placeholder: to be completed)_

---

> The scientific state of this project — what was tried, what failed and why — lives in
> `ATTACK_LADDER.md`, `RESULTS.md` and `ledger/`. That is the real asset. The weights are
> reproducible; the reasoning is not.
