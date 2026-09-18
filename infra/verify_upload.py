#!/usr/bin/env python3
"""Verifica por checksum que lo subido al volumen sea idéntico al archivo local.

Comparar tamaños detecta un archivo truncado, pero no un byte cambiado a media
transferencia. El gateway S3 de RunPod devuelve ETags con la convención
multipart de S3 —`<md5 de los md5 de cada parte>-<n partes>`— así que el
checksum se puede recalcular en local y comparar de verdad.

    python infra/verify_upload.py --prefix raw/Task2/ --local "data/Task 2"
    python infra/verify_upload.py --prefix raw/ --local data --recursive

Salidas posibles por archivo:
    OK        el checksum coincide: el archivo remoto es idéntico
    DISTINTO  llegó completo pero con contenido diferente -> hay que resubir
    TAMAÑO    el tamaño no coincide -> truncado, hay que resubir
    FALTA     no está en el volumen
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import boto3
except ImportError:
    sys.exit("boto3 no instalado. Corre:  pip install boto3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNK_MB = 8  # debe coincidir con el usado al subir


def load_env(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def multipart_etag(path: Path, chunk_bytes: int) -> str:
    """Recalcula el ETag multipart de S3 para un archivo local.

    S3 hashea cada parte por separado, concatena esos digests binarios, y hashea
    el resultado. El sufijo es el número de partes.
    """
    digests = []
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digests.append(hashlib.md5(chunk).digest())

    if len(digests) == 1:
        return digests_to_plain_md5(path)
    combined = hashlib.md5(b"".join(digests)).hexdigest()
    return f"{combined}-{len(digests)}"


def digests_to_plain_md5(path: Path) -> str:
    """MD5 del archivo entero, leído por bloques para no cargarlo en memoria."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 24)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="raw/Task2/", help="Prefijo remoto a verificar")
    parser.add_argument("--local", type=Path, required=True, help="Carpeta local correspondiente")
    parser.add_argument("--pattern", default="*.zip", help="Qué archivos comparar (default: *.zip)")
    parser.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_MB,
                        help=f"Tamaño de parte usado al subir (default: {DEFAULT_CHUNK_MB})")
    parser.add_argument("--quick", action="store_true",
                        help="Comparar solo tamaños, sin leer los archivos")
    args = parser.parse_args()

    env = load_env(PROJECT_ROOT / ".env")
    client = boto3.client(
        "s3",
        endpoint_url=env["RUNPOD_S3_ENDPOINT"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env.get("RUNPOD_S3_REGION", "EU-RO-1"),
    )
    bucket = env["RUNPOD_S3_BUCKET"]

    remote = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            remote[obj["Key"]] = (obj["Size"], obj.get("ETag", "").strip('"'))

    files = sorted(args.local.glob(args.pattern))
    if not files:
        sys.exit(f"No hay archivos {args.pattern} en {args.local}")

    chunk_bytes = args.chunk_mb * 1024 * 1024
    verdicts = {"OK": [], "DISTINTO": [], "TAMAÑO": [], "FALTA": []}

    print(f"Verificando {len(files)} archivo(s) contra s3://{bucket}/{args.prefix}")
    print(f"Modo: {'solo tamaño' if args.quick else f'checksum multipart, partes de {args.chunk_mb} MB'}\n")

    for path in files:
        key = args.prefix + path.name
        size = path.stat().st_size

        if key not in remote:
            verdicts["FALTA"].append(path.name)
            print(f"  FALTA     {path.name}  ({human(size)})")
            continue

        remote_size, remote_etag = remote[key]
        if remote_size != size:
            verdicts["TAMAÑO"].append(path.name)
            print(f"  TAMAÑO    {path.name}  local={size:,}  remoto={remote_size:,}")
            continue

        if args.quick:
            verdicts["OK"].append(path.name)
            print(f"  OK        {path.name}  ({human(size)}, solo tamaño)")
            continue

        # El ETag remoto dice qué comparar. Si trae sufijo "-N" es un hash
        # compuesto por partes; si no, el gateway consolidó el objeto y guardó
        # un MD5 del archivo completo. RunPod hace lo segundo a veces, y
        # compararlo contra el compuesto da un falso "corrupto".
        if "-" in remote_etag:
            local_etag = multipart_etag(path, chunk_bytes)
            kind = "multipart"
        else:
            local_etag = digests_to_plain_md5(path)
            kind = "md5 completo"

        if local_etag == remote_etag:
            verdicts["OK"].append(path.name)
            print(f"  OK        {path.name}  ({human(size)})  {remote_etag}  [{kind}]")
        else:
            verdicts["DISTINTO"].append(path.name)
            print(f"  DISTINTO  {path.name}  [{kind}]")
            print(f"            local  {local_etag}")
            print(f"            remoto {remote_etag}")

    print(f"\n  OK={len(verdicts['OK'])}  DISTINTO={len(verdicts['DISTINTO'])}"
          f"  TAMAÑO={len(verdicts['TAMAÑO'])}  FALTA={len(verdicts['FALTA'])}")

    pending = verdicts["DISTINTO"] + verdicts["TAMAÑO"] + verdicts["FALTA"]
    if pending:
        print("\n  Hay que (re)subir:")
        for name in pending:
            print(f"    {name}")
        sys.exit(1)
    print("\n  Todo íntegro.")


if __name__ == "__main__":
    main()
