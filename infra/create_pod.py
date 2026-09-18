#!/usr/bin/env python3
"""Crea un pod de RunPod con el volumen del proyecto montado.

La API key del proyecto es restringida: la API GraphQL responde 403 y el REST v1
no expone catálogo de GPUs, así que no hay forma de consultar disponibilidad.
Este script recorre una lista de tipos por orden de preferencia y se queda con
el primero que RunPod acepte.

El pod DEBE quedar en EU-RO-1, que es donde vive el network volume; un pod en
otro datacenter no puede montarlo.

    python infra/create_pod.py --name fido-dev
    python infra/create_pod.py --name fido-dev --gpu "NVIDIA H100 80GB HBM3"
    python infra/create_pod.py --list
    python infra/create_pod.py --terminate <pod_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API = "https://rest.runpod.io/v1"

# Orden de preferencia. La 5090 es la que pidió el usuario; las demás son
# alternativas si no hay disponibilidad, priorizando arquitecturas que soportan
# CUDA 12.8 de forma nativa.
GPU_PREFERENCE = [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA B200",
    "NVIDIA H200",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
]

# Imagen con CUDA 12.8: obligatorio para Blackwell (sm_120), y además es lo que
# usa el evaluador de Codabench (pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime).
DEFAULT_IMAGE = "runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204"


def load_env() -> dict:
    values = {}
    for line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def public_keys() -> str:
    """Claves públicas locales, una por línea, para authorized_keys del pod.

    Recoge todas las .pub de ~/.ssh que tengan su privada al lado: no sirve
    autorizar una clave cuya privada no tenemos.
    """
    ssh_dir = Path.home() / ".ssh"
    keys = []
    if ssh_dir.is_dir():
        for pub in sorted(ssh_dir.glob("*.pub")):
            if pub.with_suffix("").exists():          # existe la privada
                keys.append(pub.read_text(encoding="utf-8").strip())
    if not keys:
        sys.exit("No hay pares de claves SSH en ~/.ssh; el pod nacería inaccesible.\n"
                 "Genera una con:  ssh-keygen -t ed25519 -N \"\" -f ~/.ssh/id_ed25519")
    return "\n".join(keys)


def call(method: str, path: str, key: str, payload=None):
    request = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8", "replace")
            return response.status, (json.loads(body) if body.strip() else None)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:500]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fido-dev")
    parser.add_argument("--gpu", help="Forzar un tipo de GPU en vez de recorrer la lista")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk-gb", type=int, default=60)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--terminate", metavar="POD_ID")
    args = parser.parse_args()

    env = load_env()
    key = env["RUNPOD_API_KEY"]
    volume_id = env["RUNPOD_S3_BUCKET"]

    if args.terminate:
        status, body = call("DELETE", f"/pods/{args.terminate}", key)
        print(f"  DELETE {args.terminate} -> {status} {body if body else ''}")
        return

    if args.list:
        status, pods = call("GET", "/pods", key)
        if not pods:
            print("  (no hay pods)")
            return
        for pod in pods:
            gpu = (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpuTypeIds")
            print(f"  {pod['id']}  {pod.get('name'):<20} {pod.get('desiredStatus'):<10} "
                  f"{gpu}  costo/h={pod.get('costPerHr')}")
            for port in (pod.get("portMappings") or {}).items():
                print(f"      puerto {port}")
        return

    candidates = [args.gpu] if args.gpu else GPU_PREFERENCE

    for gpu_type in candidates:
        payload = {
            "name": args.name,
            "imageName": args.image,
            "gpuTypeIds": [gpu_type],
            "gpuCount": args.gpu_count,
            "cloudType": "SECURE",
            "computeType": "GPU",
            # El volumen vive en EU-RO-1; el pod tiene que nacer ahí para montarlo.
            "networkVolumeId": volume_id,
            "dataCenterIds": ["EU-RO-1"],
            "volumeMountPath": "/workspace",
            "containerDiskInGb": args.disk_gb,
            "ports": ["8888/http", "22/tcp"],
            # Sin esto RunPod inyecta solo la clave ed25519 de la cuenta, cuya
            # privada no está en esta máquina: el pod nace inaccesible por SSH.
            # Pasamos explícitamente las claves públicas locales.
            "env": {"JUPYTER_PASSWORD": "", "PUBLIC_KEY": public_keys()},
        }
        print(f"  probando {gpu_type} ...", end=" ", flush=True)
        status, body = call("POST", "/pods", key, payload)
        if status in (200, 201):
            print("CREADO")
            print(json.dumps(body, indent=2)[:1500])
            pod_id = body.get("id") if isinstance(body, dict) else None
            if pod_id:
                print(f"\n  POD_ID={pod_id}")
                print(f"  Jupyter (cuando arranque): https://{pod_id}-8888.proxy.runpod.net")
            return
        print(f"no ({status})")
        if isinstance(body, str) and "no longer any instances" not in body.lower():
            print(f"      {body[:220]}")

    sys.exit("Ningún tipo de GPU disponible en EU-RO-1 con estos parámetros.")


if __name__ == "__main__":
    main()
