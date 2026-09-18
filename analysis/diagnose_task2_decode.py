#!/usr/bin/env python3
"""¿El heatmap de T2 sabe dónde está la plantilla, y lo estamos leyendo mal?

El diagnóstico de componentes (diagnose_task2_error.py) mostró que el 65.6% del
error de esquina viene de la posición. Hay dos explicaciones posibles y exigen
arreglos opuestos:

  (a) el heatmap NO sabe -> el pico está en el lugar equivocado; hay que
      entrenar mejor la correlación.
  (b) el heatmap SÍ sabe pero lo leemos mal -> el soft-argmax GLOBAL promedia
      todo el mapa; si el pico no domina, la predicción se arrastra hacia el
      centro. Se arregla decodificando distinto, sin reentrenar.

Este script las separa comparando cuatro decodificadores sobre el MISMO modelo
entrenado, contra el centro real derivado del GT.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fido.data.common import group_kfold_indices  # noqa: E402
from fido.data.task2 import Task2Dataset  # noqa: E402
from fido.heatmap_decode import local_soft_argmax_2d, soft_argmax_2d  # noqa: E402
from fido.models.task2_baseline import SCALE_REF, FundusEnfaceHeatmapModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--enface-cache", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    dataset = Task2Dataset(args.root, include_vessel_enface=False,
                           include_vessel_mask=False,
                           enface_cache_dir=args.enface_cache)
    groups = [c["scenario"] for c in dataset.cases]
    kfold = group_kfold_indices(groups, n_splits=args.n_folds, seed=0)
    for _ in range(args.fold + 1):
        train_idx, val_idx = next(kfold)
    loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    model = FundusEnfaceHeatmapModel(base_channels=32, n_downsamples=4)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device).eval()

    # Se reimplementa el forward hasta el heatmap para poder probar varios
    # decodificadores sobre exactamente el mismo mapa.
    errores = {"soft global (actual)": [], "argmax duro": [],
               "soft local w=5": [], "soft global T=0.1": []}
    pico_en_celda_correcta = []
    concentracion = []

    with torch.no_grad():
        for batch in loader:
            fundus = batch["fundus"].to(args.device)
            enface = batch["enface"].to(args.device)
            gt = batch["gt_matrix"].to(args.device).double()

            enface_r = F.interpolate(enface, size=(int(SCALE_REF), int(SCALE_REF)),
                                     mode="bilinear", align_corners=False)
            f_feat = model.fundus_encoder(fundus)
            e_feat = model.enface_encoder(enface_r)
            B, C, Hf, Wf = f_feat.shape
            _, _, Ht, Wt = e_feat.shape
            corr = F.conv2d(f_feat.reshape(1, B * C, Hf, Wf), e_feat,
                            groups=B, padding=(Ht // 2, Wt // 2))
            hm = corr.reshape(B, 1, Hf + 2 * (Ht // 2) - Ht + 1,
                              Wf + 2 * (Wt // 2) - Wt + 1)
            hm = hm / (C ** 0.5)
            hm = (hm - hm.mean(dim=(2, 3), keepdim=True)) / (hm.std(dim=(2, 3), keepdim=True) + 1e-6)

            stride = 1024.0 / Hf
            # Centro real del GT, en celdas del heatmap (mismo sistema que la
            # predicción: +0.5 por el kernel de lado par).
            half = torch.full((2,), 0.5, device=gt.device, dtype=gt.dtype)
            centro_gt = (gt[:, :2, :2] @ half + gt[:, :2, 2]) / stride + 0.5

            decs = {
                "soft global (actual)": soft_argmax_2d(hm, temperature=1.0),
                "soft local w=5": local_soft_argmax_2d(hm, window=5, temperature=1.0),
                "soft global T=0.1": soft_argmax_2d(hm, temperature=0.1),
            }
            flat = hm.reshape(B, -1)
            idx = flat.argmax(dim=-1)
            duro = torch.stack([(idx % hm.shape[-1]).float(),
                                (idx // hm.shape[-1]).float()], dim=-1).unsqueeze(1)
            decs["argmax duro"] = duro

            for nombre, coords in decs.items():
                xy = coords[:, 0, :].double()
                err = torch.linalg.norm(xy - centro_gt, dim=-1) * stride
                errores[nombre].extend(err.cpu().numpy().tolist())

            # ¿El pico duro cae en la celda correcta (o pegado a ella)?
            d_celdas = torch.linalg.norm(duro[:, 0, :].double() - centro_gt, dim=-1)
            pico_en_celda_correcta.extend((d_celdas <= 1.5).cpu().numpy().tolist())
            # Concentración: qué fracción de la masa softmax se lleva el pico.
            p = torch.softmax(flat, dim=-1)
            concentracion.extend(p.max(dim=-1).values.cpu().numpy().tolist())

    n = len(concentracion)
    print(f"\nError de LOCALIZACIÓN del centro, {n} casos de validación\n")
    print(f"{'decodificador':24s} {'media':>9s} {'mediana':>9s} {'<16px':>8s}")
    for nombre, vals in errores.items():
        v = np.array(vals)
        print(f"{nombre:24s} {v.mean():8.2f}px {np.median(v):8.2f}px "
              f"{(v < 16).mean()*100:7.1f}%")

    print(f"\nPico duro dentro de 1.5 celdas del centro real: "
          f"{np.mean(pico_en_celda_correcta)*100:.1f}% de los casos")
    print(f"Masa del softmax que se lleva el pico: mediana {np.median(concentracion)*100:.3f}% "
          f"(1/{hm.shape[-1]*hm.shape[-2]} = {100/(hm.shape[-1]*hm.shape[-2]):.3f}% sería uniforme)")
    print("\nLectura: si 'argmax duro' es MUCHO mejor que 'soft global', el mapa")
    print("sabe dónde está y el decodificador lo está arruinando (se arregla sin")
    print("reentrenar). Si los dos son igual de malos, el problema es el mapa.")


if __name__ == "__main__":
    main()
