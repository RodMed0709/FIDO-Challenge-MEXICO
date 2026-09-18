#!/usr/bin/env python3
"""Ensambla el explicador de Task 2: mete los SVG y las imagenes en la plantilla.

Las figuras fotograficas van incrustadas como data URI porque la CSP de los
artifacts bloquea cualquier peticion a un host externo.

    python analysis/render_task2_figures.py     # primero, genera los JPEG
    python analysis/build_task2_explainer.py    # despues, arma el HTML
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = PROJECT_ROOT / "docs"


def rotated_square(cx: float, cy: float, half: float, degrees: float) -> list[tuple[float, float]]:
    """Esquinas de un cuadrado rotado, en orden."""
    radians = math.radians(degrees)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    corners = []
    for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
        corners.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return corners


def points(corners) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in corners)


# --------------------------------------------------------------------------
# Figura A: el mapeo de coordenadas
# --------------------------------------------------------------------------
def svg_mapping() -> str:
    left = rotated_square(150, 170, 80, 0)
    right = rotated_square(565, 178, 58, 20)

    corner_labels = [
        (left[0][0] - 6, left[0][1] - 10, "(-1,-1)", "end"),
        (left[1][0] + 6, left[1][1] - 10, "(1,-1)", "start"),
        (left[2][0] + 6, left[2][1] + 18, "(1,1)", "start"),
        (left[3][0] - 6, left[3][1] + 18, "(-1,1)", "end"),
    ]
    labels = "".join(
        f'<text class="dg-lbl" x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}">{text}</text>'
        for x, y, text, anchor in corner_labels
    )
    dots_left = "".join(f'<circle class="dg-acc" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>' for x, y in left)
    dots_right = "".join(f'<circle class="dg-acc" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>' for x, y in right)

    return f'''<svg class="diagram" viewBox="0 0 760 340" role="img"
     aria-label="La matriz M lleva el cuadrado normalizado del volumen iOCT, de menos uno a uno, hasta un cuadrado rotado dentro de la imagen del microscopio medida en pixeles.">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
    </marker>
  </defs>

  <text class="dg-lbl" x="150" y="26" text-anchor="middle">VOLUMEN iOCT</text>
  <text class="dg-txt" x="150" y="46" text-anchor="middle" opacity=".7">normalizado &#8722;1 a 1</text>
  <line class="dg-line" x1="40" y1="170" x2="260" y2="170"/>
  <line class="dg-line" x1="150" y1="60" x2="150" y2="280"/>
  <polygon class="dg-sq" points="{points(left)}"/>
  {dots_left}{labels}
  <circle cx="150" cy="170" r="4" fill="currentColor"/>
  <text class="dg-lbl" x="150" y="300" text-anchor="middle">origen al centro</text>

  <line class="dg-arrow" x1="275" y1="170" x2="420" y2="170" marker-end="url(#ar)"/>
  <text class="dg-txt" x="347" y="158" text-anchor="middle" font-style="italic">M</text>
  <text class="dg-lbl" x="347" y="192" text-anchor="middle">la matriz 3&#215;3</text>

  <text class="dg-lbl" x="570" y="26" text-anchor="middle">IMAGEN DEL MICROSCOPIO</text>
  <text class="dg-txt" x="570" y="46" text-anchor="middle" opacity=".7">p&#237;xeles 0 a 1024</text>
  <rect class="dg-frame" x="450" y="60" width="240" height="240" rx="2"/>
  <text class="dg-lbl" x="454" y="76">0,0</text>
  <text class="dg-lbl" x="686" y="292" text-anchor="end">1024,1024</text>
  <polygon class="dg-sq" points="{points(right)}"/>
  {dots_right}
  <circle cx="565" cy="178" r="4" fill="currentColor"/>
  <text class="dg-lbl" x="570" y="322" text-anchor="middle">d&#243;nde cay&#243; el escaneo</text>
</svg>'''


# --------------------------------------------------------------------------
# Figura B: las cuatro perillas
# --------------------------------------------------------------------------
def svg_knobs() -> str:
    panels = []
    titles = ["POSICI&#211;N", "TAMA&#209;O", "&#193;NGULO", "ESPEJO"]
    subtitles = ["tx, ty", "s", "&#952;", "fijo"]

    for index in range(4):
        ox = index * 190
        cx, cy = ox + 95, 108
        base = rotated_square(cx, cy, 40, 0)
        ghost = f'<polygon class="dg-ghost" points="{points(base)}"/>'

        if index == 0:
            moved = rotated_square(cx + 22, cy - 16, 40, 0)
            extra = (f'<line class="dg-arrow" x1="{cx}" y1="{cy}" x2="{cx + 20}" y2="{cy - 14}" '
                     f'marker-end="url(#ar2)"/>')
        elif index == 1:
            moved = rotated_square(cx, cy, 55, 0)
            extra = (f'<line class="dg-arrow" x1="{cx + 40}" y1="{cy}" x2="{cx + 53}" y2="{cy}" '
                     f'marker-end="url(#ar2)"/>')
        elif index == 2:
            moved = rotated_square(cx, cy, 40, 30)
            extra = (f'<path class="dg-arrow" d="M {cx + 52} {cy} A 52 52 0 0 1 {cx + 45.0:.1f} '
                     f'{cy + 26.0:.1f}" marker-end="url(#ar2)"/>')
        else:
            moved = base
            # Una marca asimetrica es lo unico que deja ver un espejo.
            extra = (f'<polyline class="dg-mark" points="{cx - 22},{cy - 20} {cx - 22},{cy + 16} '
                     f'{cx - 2},{cy + 16}"/>'
                     f'<polyline class="dg-mark-flip" points="{cx + 22},{cy - 20} {cx + 22},{cy + 16} '
                     f'{cx + 2},{cy + 16}"/>')
            ghost = ""

        panels.append(
            f'<text class="dg-lbl" x="{cx}" y="30" text-anchor="middle">{titles[index]}</text>'
            f'<text class="dg-txt dg-acc-fill" x="{cx}" y="50" text-anchor="middle" '
            f'font-style="italic">{subtitles[index]}</text>'
            f'{ghost}<polygon class="dg-sq" points="{points(moved)}"/>{extra}'
        )

    dividers = "".join(
        f'<line class="dg-div" x1="{i * 190 + 190}" y1="20" x2="{i * 190 + 190}" y2="185"/>'
        for i in range(3)
    )

    return f'''<svg class="diagram" viewBox="0 0 760 200" role="img"
     aria-label="Cuatro paneles que muestran lo que cambia cada parametro: la posicion del cuadrado, su tamano, su angulo, y el espejo que siempre esta puesto.">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  {dividers}
  {"".join(panels)}
</svg>'''


# --------------------------------------------------------------------------
# Figura C: de donde sale el error
# --------------------------------------------------------------------------
def svg_error() -> str:
    truth = [(70, 90), (230, 90), (230, 250), (70, 250)]
    predicted = [(96, 106), (252, 118), (240, 276), (84, 264)]
    segments = "".join(
        f'<line class="dg-err" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>'
        for a, b in zip(truth, predicted)
    )
    truth_dots = "".join(f'<circle class="dg-acc" cx="{x}" cy="{y}" r="4"/>' for x, y in truth)
    pred_dots = "".join(f'<circle class="dg-err-fill" cx="{x}" cy="{y}" r="4"/>' for x, y in predicted)

    corner_names = [
        (62, 84, "(0,0)", "end"), (238, 84, "(1,0)", "start"),
        (238, 264, "(1,1)", "start"), (62, 264, "(0,1)", "end"),
    ]
    names = "".join(
        f'<text class="dg-lbl" x="{x}" y="{y}" text-anchor="{a}">{t}</text>'
        for x, y, t, a in corner_names
    )

    # Panel derecho: que esquinas mueve cada parametro.
    rows = [
        ("centro (tx, ty)", [True, True, True, True], "4 de 4"),
        ("columna 0", [False, True, True, False], "2 de 4"),
        ("columna 1", [False, False, True, True], "2 de 4"),
    ]
    right = []
    for row_index, (label, mask, count) in enumerate(rows):
        y0 = 80 + row_index * 74
        square = [(430, y0), (486, y0), (486, y0 + 56), (430, y0 + 56)]
        dots = "".join(
            f'<circle class="{"dg-err-fill" if on else "dg-off"}" cx="{x}" cy="{y}" r="5"/>'
            for (x, y), on in zip(square, mask)
        )
        right.append(
            f'<polygon class="dg-ghost" points="{points(square)}"/>{dots}'
            f'<text class="dg-txt" x="516" y="{y0 + 26}">{label}</text>'
            f'<text class="dg-lbl dg-err-fill" x="516" y="{y0 + 46}">{count}</text>'
        )

    return f'''<svg class="diagram" viewBox="0 0 760 320" role="img"
     aria-label="A la izquierda, el cuadrado verdadero y el predicho con los cuatro segmentos de error entre sus esquinas. A la derecha, cuantas esquinas mueve cada parametro: el centro mueve las cuatro, cada columna lineal solo dos.">
  <text class="dg-lbl" x="150" y="42" text-anchor="middle">EL ERROR DE ESQUINAS</text>
  <polygon class="dg-sq" points="{points(truth)}"/>
  <polygon class="dg-pred" points="{points(predicted)}"/>
  {segments}{truth_dots}{pred_dots}{names}
  <text class="dg-lbl dg-acc-fill" x="150" y="296" text-anchor="middle">verdadero</text>
  <text class="dg-lbl dg-err-fill" x="150" y="312" text-anchor="middle">predicho</text>

  <line class="dg-div" x1="360" y1="30" x2="360" y2="290"/>

  <text class="dg-lbl" x="430" y="42">QU&#201; MUEVE CADA PAR&#193;METRO</text>
  {"".join(right)}
</svg>'''


def main():
    template = (DOCS / "task2_explainer_template.html").read_text(encoding="utf-8")
    figures = json.loads((DOCS / "figures" / "figures_b64.json").read_text(encoding="utf-8"))

    replacements = {
        "{{SVG_MAPPING}}": svg_mapping(),
        "{{SVG_KNOBS}}": svg_knobs(),
        "{{SVG_ERROR}}": svg_error(),
        "{{FIG_FUNDUS}}": figures["fig_fundus"],
        "{{FIG_CROSSHAIR}}": figures["fig_crosshair"],
        "{{FIG_ENFACE}}": figures["fig_enface"],
        "{{FIG_OVERLAY}}": figures["fig_overlay"],
    }
    for token, value in replacements.items():
        if token not in template:
            raise SystemExit(f"La plantilla no contiene {token}")
        template = template.replace(token, value)

    out = DOCS / "task2_explainer.html"
    out.write_text(template, encoding="utf-8")
    print(f"  {out}   {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
