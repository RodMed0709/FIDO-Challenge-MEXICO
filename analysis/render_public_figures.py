"""Renderiza las figuras del README público, en variante clara y oscura.

Todos los números son literales tomados de los resultados medidos de este repo;
cada bloque cita su fuente. El script no toca el dataset: las figuras del repo
público no pueden contener píxeles del challenge (CC BY-NC-ND), así que aquí solo
se dibujan nuestras propias mediciones.

    python analysis/render_public_figures.py --out docs/figures_public

Paleta: slots categóricos 1 (azul) y 2 (naranja) del sistema por defecto, más la
paleta de estado para el eje aceptado/rechazado. Validada con
`scripts/validate_palette.js` en ambos modos: todas las comprobaciones pasan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series_1: str
    series_2: str
    critical: str = "#d03b3b"
    good: str = "#0ca30c"


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781",
              "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834")
DARK = Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781",
             "#2c2c2a", "#383835", "#3987e5", "#d95926")


def grid(ax, t: Theme, axis: str):
    ax.grid(True, axis=axis, color=t.grid, linewidth=1.0, alpha=1.0)


def new_axes(t: Theme, figsize, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=t.surface)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(t.surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t.axis)
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=t.muted, labelsize=9, length=3, width=1.0)
        # La rejilla la enciende cada figura por eje; por defecto, ninguna.
        ax.grid(False)
        ax.set_axisbelow(True)
    return fig, axes


def finish(fig, t: Theme, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{t.name}.png"
    fig.savefig(path, dpi=160, facecolor=t.surface, bbox_inches="tight",
                pad_inches=0.28)
    plt.close(fig)
    return path


def title(ax, t: Theme, head: str, sub: str | None = None):
    """Título y subtítulo por encima del área de dibujo, sin solaparse.

    `ax.set_title` no reserva espacio para un subtítulo de varias líneas, así que
    ambos se colocan a mano: el subtítulo ancla su base justo encima del eje y el
    título se levanta tantas líneas como ocupe aquél.
    """
    gap = 0.052  # altura de una línea de subtítulo, en fracción del eje
    if sub:
        lines = sub.count("\n") + 1
        ax.text(0.0, 1.03, sub, transform=ax.transAxes, color=t.secondary,
                fontsize=9.5, va="bottom", ha="left", linespacing=1.45)
        head_y = 1.03 + lines * gap + 0.030
    else:
        head_y = 1.03
    ax.text(0.0, head_y, head, transform=ax.transAxes, color=t.text,
            fontsize=13, fontweight="600", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# 1. El reescalado de la distancia de Task 1
#     Fuente: submissions/r05-dist-rescaled/README.md, NOW.md:80-92, RESULTS.md:229
# ---------------------------------------------------------------------------
def fig_distance_rescale(t: Theme, out: Path):
    gap = np.linspace(13, 410, 400)
    old = 0.7320 * gap + 2.6779
    new = 0.9370 * gap + 3.4277

    fig, ax = new_axes(t, (7.6, 4.6))
    grid(ax, t, "both")
    ax.plot(gap, new, color=t.series_1, linewidth=2.0,
            label="corrected  0.9370·gap + 3.4277")
    ax.plot(gap, old, color=t.series_2, linewidth=2.0,
            label="as shipped in r01–r04  0.7320·gap + 2.6779")

    # El caso citado: verdad 78.69 px, predicción vieja 61.38 px.
    g0 = (78.69 - 3.4277) / 0.9370
    ax.plot([g0, g0], [0.7320 * g0 + 2.6779, 78.69], color=t.critical,
            linewidth=2.0, solid_capstyle="round", zorder=5)
    ax.plot([g0], [78.69], "o", color=t.series_1, markersize=8,
            markeredgecolor=t.surface, markeredgewidth=2, zorder=6)
    ax.plot([g0], [0.7320 * g0 + 2.6779], "o", color=t.series_2, markersize=8,
            markeredgecolor=t.surface, markeredgewidth=2, zorder=6)
    ax.annotate("17.3 px of systematic bias\nagainst a 20 px threshold",
                xy=(g0, 70.0), xytext=(g0 + 115, 36.0),
                color=t.critical, fontsize=9.5, fontweight="600",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=t.critical,
                                linewidth=1.2, shrinkA=6, shrinkB=4))

    title(ax, t, "One constant was the whole Task 1 distance gap",
          "Predicted tool–tissue distance. Ground truth is stored scaled; the scorer's divisor "
          "changed from 10 to 7.8125,\nmaking the real target 1.28× larger. Rescaling the fitted "
          "constants needed no retraining.")
    ax.set_xlabel("pixel gap between instrument tip and retina", color=t.secondary, fontsize=10)
    ax.set_ylabel("predicted distance (px)", color=t.secondary, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for text in leg.get_texts():
        text.set_color(t.secondary)
    ax.text(1.0, -0.17, "fit measured over 92,012 measurements across 61,691 frames, R² = 0.9916",
            transform=ax.transAxes, color=t.muted, fontsize=8.5, ha="right")
    return finish(fig, t, out, "01-distance-rescale")


# ---------------------------------------------------------------------------
# 2. De dónde sale el hueco con el líder
#     Fuente: RESULTS.md:54,243 · HANDOFF_2026-08-18.md:27,45 ·
#             experiments/101-t1-tta-ensemble/INFORME.md:5,8
# ---------------------------------------------------------------------------
def fig_score_decomposition(t: Theme, out: Path):
    rows = [
        ("r01  ours", 0.7309, 0.0900, False),
        ("r06  ours, best", 0.7573, 0.12997, True),
        ("leader", 0.7973, 0.2027, False),
    ]
    labels = [r[0] for r in rows]
    kp = np.array([0.7 * r[1] for r in rows])
    dist = np.array([0.3 * r[2] for r in rows])
    y = np.arange(len(rows))[::-1]

    fig, ax = new_axes(t, (7.8, 3.6))
    grid(ax, t, "x")
    ax.barh(y, kp, height=0.40, color=t.series_1, label="0.7 × keypoint AUC")
    ax.barh(y, dist, height=0.40, left=kp + 0.004, color=t.series_2,
            label="0.3 × distance AUC")

    for yi, (lab, k, d, derived) in zip(y, rows):
        ax.text(0.7 * k / 2, yi, f"keypoint {k:.4f}", color="#ffffff",
                fontsize=9.5, fontweight="600", va="center", ha="center")
        mark = "*" if derived else ""
        end = 0.7 * k + 0.3 * d
        ax.text(end + 0.016, yi, f"{end:.4f}", color=t.text, fontsize=10.5,
                fontweight="600", va="center")
        # Columna alineada a la derecha: los tres valores de distancia se leen
        # como una columna, no pegados al final de cada barra.
        ax.text(0.985, yi, f"distance {d:.4f}{mark}", color=t.secondary,
                fontsize=9, va="center", ha="right")

    ax.set_yticks(y, labels, color=t.text, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    title(ax, t, "The gap to the leader was never in the keypoint",
          "Task 1 final score = 0.7 × keypoint AUC + 0.3 × distance AUC. "
          "Our keypoint trails by 0.04;\nour distance trails by 0.07 of a component "
          "that is worth 0.30.")
    leg = ax.legend(frameon=False, fontsize=9.5, ncols=2, loc="upper left",
                    bbox_to_anchor=(0.0, -0.13))
    for text in leg.get_texts():
        text.set_color(t.secondary)
    ax.text(1.0, -0.30, "* r06's distance AUC is derived from its published total and keypoint AUC, "
            "not reported directly",
            transform=ax.transAxes, color=t.muted, fontsize=8.5, ha="right")
    return finish(fig, t, out, "02-score-decomposition")


# ---------------------------------------------------------------------------
# 3. El colapso del TTA
#     Fuente: commit 21556db
# ---------------------------------------------------------------------------
def fig_tta_collapse(t: Theme, out: Path):
    rows = [
        ("baseline, no TTA", 0.8697, 0.91, True),
        ("ensemble of 3, no TTA", 0.8697, 0.88, True),
        ("TTA multi-scale, 3 views", 0.8606, 1.02, True),
        ("TTA combined, 8 views", 0.0455, 38.60, False),
        ("ensemble + TTA", 0.0485, 35.19, False),
        ("TTA flips + rotations, 6 views", 0.0061, 70.62, False),
    ]
    labels = [r[0] for r in rows]
    auc = np.array([r[1] for r in rows])
    err = [r[2] for r in rows]
    keeps = [r[3] for r in rows]
    y = np.arange(len(rows))[::-1]
    colors = [t.series_1 if k else t.critical for k in keeps]

    fig, ax = new_axes(t, (7.8, 4.2))
    grid(ax, t, "x")
    ax.barh(y, auc, height=0.56, color=colors)
    for yi, a, e, k in zip(y, auc, err, keeps):
        ax.text(a + 0.012, yi, f"{a:.4f}", color=t.text, fontsize=10,
                fontweight="600", va="center")
        ax.text(1.30, yi, f"mean error {e:>6.2f} px", color=t.secondary,
                fontsize=9, va="center", ha="right")
    ax.set_yticks(y, labels, color=t.text, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.32)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("keypoint AUC", color=t.secondary, fontsize=10)
    title(ax, t, "Test-time augmentation does not underperform — it destroys the model",
          "30 held-out cases, same GroupKFold split. Flips and rotations are the culprit; "
          "multi-scale is harmless.\nNot a de-transform bug: 45 synthetic tests recover the "
          "coordinate to ~0 px. It is anatomy — the retina has\na preferred orientation, so a "
          "flipped fundus is an anatomically impossible image.")
    ax.text(1.0, -0.20, "red = rejected against a pre-registered +0.01 gate",
            transform=ax.transAxes, color=t.muted, fontsize=8.5, ha="right")
    return finish(fig, t, out, "03-tta-collapse")


# ---------------------------------------------------------------------------
# 4. Augmentación de escala en Task 2
#     Fuente: experiments/81-t2-scale-augmentation/holdout_results.json
# ---------------------------------------------------------------------------
def fig_scale_augmentation(t: Theme, out: Path):
    panels = [
        ("corner AUC", 0.0005498533724340175, 0.006048387096774193, "{:.4f}"),
        ("mean corner error (px)", 85.76509462683644, 65.30496928035546, "{:.1f}"),
        ("scale MAE (px)", 31.162982593864125, 14.277628194702732, "{:.1f}"),
    ]
    fig, axes = new_axes(t, (8.4, 3.4), 1, 3)
    for ax, (name, base, aug, fmt) in zip(axes, panels):
        # Cada panel tiene su propia escala y los valores van etiquetados:
        # la rejilla no aporta y las verticales se leían como separadores falsos.
        ax.bar([0, 1], [base, aug], width=0.56,
               color=[t.muted, t.series_1])
        top = max(base, aug)
        for x, v in zip([0, 1], [base, aug]):
            ax.text(x, v + top * 0.05, fmt.format(v), color=t.text,
                    fontsize=10, fontweight="600", ha="center")
        ax.set_ylim(0, top * 1.28)
        ax.set_xticks([0, 1], ["baseline", "augmented"], color=t.text, fontsize=9.5)
        ax.tick_params(axis="x", length=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(name, color=t.secondary, fontsize=10, labelpad=8)

    fig.text(0.0, 1.20, "Exact geometric scale augmentation, as the only change",
             color=t.text, fontsize=13, fontweight="600", ha="left", va="bottom")
    fig.text(0.0, 1.01, "Synthetic holdout, 496 evaluations. Scale error falls 54.2% against a "
             "pre-registered 20% gate,\nand no scenario regresses. Corner AUC rises 11×, and is "
             "still 0.006 — the ceiling was elsewhere.",
             color=t.secondary, fontsize=9.5, ha="left", va="bottom", linespacing=1.45)
    return finish(fig, t, out, "04-scale-augmentation")


FIGURES = (fig_distance_rescale, fig_score_decomposition, fig_tta_collapse,
           fig_scale_augmentation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/figures_public"))
    args = parser.parse_args()
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    for theme in (LIGHT, DARK):
        for fn in FIGURES:
            print(fn(theme, args.out))



# ---------------------------------------------------------------------------
# 5. Esquema de la geometría de Task 2 (SVG, dibujado — sin píxeles del dataset)
#     Fuente: RULES_OF_ENGAGEMENT.md:110-125 · THE_MAP.md:171-179
# ---------------------------------------------------------------------------
def _quad(cx, cy, s, deg, flip=True):
    """Cuatro esquinas del cuadrado unitario bajo una similitud reflejada."""
    r = np.deg2rad(deg)
    a, b = s * np.cos(r), s * np.sin(r)
    # Reflejada: determinante negativo, que es lo que cumple el GT en 1214/1214.
    m = np.array([[a, b], [b, -a]]) if flip else np.array([[a, -b], [b, a]])
    pts = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=float)
    return [(cx + x, cy + y) for x, y in pts @ m.T]


def fig_task2_geometry(t: Theme, out: Path):
    gt = _quad(600, 238, 70, 24)
    pred = _quad(612, 250, 76, 33)
    poly = lambda p: " ".join(f"{x:.1f},{y:.1f}" for x, y in p)
    names = ["(0,0)", "(1,0)", "(1,1)", "(0,1)"]

    links = "".join(
        f'<line x1="{g[0]:.1f}" y1="{g[1]:.1f}" x2="{p[0]:.1f}" y2="{p[1]:.1f}" '
        f'stroke="{t.critical}" stroke-width="2.5" stroke-dasharray="5 4"/>'
        for g, p in zip(gt, pred))
    gt_dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{t.series_1}" '
        f'stroke="{t.surface}" stroke-width="2"/>' for x, y in gt)
    pred_dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{t.series_2}" '
        f'stroke="{t.surface}" stroke-width="2"/>' for x, y in pred)
    unit_labels = "".join(
        f'<text x="{x}" y="{y}" fill="{t.muted}" font-size="13" '
        f'text-anchor="{anc}">{n}</text>'
        for (x, y, anc), n in zip(
            [(96, 326, "end"), (274, 326, "start"), (274, 142, "start"), (96, 142, "end")],
            names))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 470" width="880" height="470" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">
<rect width="880" height="470" fill="{t.surface}"/>
<text x="24" y="38" fill="{t.text}" font-size="19" font-weight="600">Task 2 is four degrees of freedom, and translation costs double</text>
<text x="24" y="63" fill="{t.secondary}" font-size="13.5">The returned 3x3 matrix maps the iOCT unit square onto microscope pixels. Verified over 1,214 snapshots: negative</text>
<text x="24" y="82" fill="{t.secondary}" font-size="13.5">determinant in 100%, orthogonal to within 10 degrees in 100% -- a reflected similarity, not a free homography.</text>

<rect x="104" y="146" width="162" height="162" fill="none" stroke="{t.axis}" stroke-width="2"/>
{unit_labels}
<text x="185" y="227" fill="{t.secondary}" font-size="13" text-anchor="middle">iOCT</text>
<text x="185" y="245" fill="{t.secondary}" font-size="13" text-anchor="middle">unit square</text>

<line x1="296" y1="227" x2="392" y2="227" stroke="{t.axis}" stroke-width="2"/>
<polygon points="392,227 382,222 382,232" fill="{t.axis}"/>
<text x="344" y="214" fill="{t.text}" font-size="13" font-weight="600" text-anchor="middle">M (3x3)</text>
<text x="344" y="250" fill="{t.muted}" font-size="12" text-anchor="middle">scale, rotation,</text>
<text x="344" y="266" fill="{t.muted}" font-size="12" text-anchor="middle">reflection, translation</text>

<rect x="430" y="116" width="340" height="250" fill="none" stroke="{t.axis}" stroke-width="2"/>
<text x="760" y="136" fill="{t.muted}" font-size="12" text-anchor="end">microscope image, 1024 x 1024 px</text>
{links}
<polygon points="{poly(gt)}" fill="none" stroke="{t.series_1}" stroke-width="2.5"/>
<polygon points="{poly(pred)}" fill="none" stroke="{t.series_2}" stroke-width="2.5"/>
{gt_dots}{pred_dots}

<circle cx="452" cy="404" r="5" fill="{t.series_1}"/>
<text x="466" y="409" fill="{t.secondary}" font-size="13">ground truth corners</text>
<circle cx="640" cy="404" r="5" fill="{t.series_2}"/>
<text x="654" y="409" fill="{t.secondary}" font-size="13">predicted corners</text>
<line x1="452" y1="432" x2="476" y2="432" stroke="{t.critical}" stroke-width="2" stroke-dasharray="4 4"/>
<text x="486" y="437" fill="{t.secondary}" font-size="13">score = AUC over thresholds of the mean of these four distances</text>
<text x="24" y="452" fill="{t.muted}" font-size="11.5">Translation moves all four corners; each linear column moves only two -- so the same error in tx,ty costs twice what it costs in scale or rotation.</text>
</svg>
'''
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"05-task2-geometry-{t.name}.svg"
    path.write_text(svg, encoding="utf-8")
    return path


FIGURES = FIGURES + (fig_task2_geometry,)

if __name__ == "__main__":
    main()
