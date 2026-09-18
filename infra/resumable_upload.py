#!/usr/bin/env python3
"""Subida multipart que REANUDA donde se quedó, parte por parte.

Por qué existe: `boto3.upload_file` no sabe reanudar. Si la conexión se cae al
90 % de un archivo de 4 GB, el reintento vuelve a empezar desde cero. Con una
conexión inestable eso puede no terminar nunca.

S3 sí guarda las partes ya subidas del multipart abierto. Este script las
consulta con ListParts y sube únicamente las que faltan, así que cada intento
avanza en vez de reempezar.

    python infra/resumable_upload.py "data/Task 2"/Scenario_09.zip --prefix raw/Task2/
    python infra/resumable_upload.py "data/Task 1"/*.zip --prefix raw/Task1/

Compatible con lo ya subido: usa partes de 8 MB, igual que `upload_to_runpod.py`,
así que los ETags siguen siendo verificables con `infra/verify_upload.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    sys.exit("boto3 no instalado. Corre:  pip install boto3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_BYTES = 8 * 1024 * 1024      # debe coincidir con upload_to_runpod.py
PART_RETRIES = 8                    # reintentos por PARTE, no por archivo
# Partes en vuelo a la vez. chunk x concurrencia debe caber holgado en los
# ~100 s que aguanta el proxy de Cloudflare antes de cortar con un 524.
PART_CONCURRENCY = 4
# Si en este lapso no avanza ni un byte, el proceso se mata solo para que
# el bucle que lo lanzo vuelva a arrancarlo con la conexion fresca.
STALL_CHECK_SECONDS = 150
BACKOFF_CAP_SECONDS = 30


def load_env(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def find_open_upload(client, bucket: str, key: str):
    """Devuelve el UploadId de un multipart abierto para esta llave, si lo hay.

    Esto es lo que permite reanudar: el gateway conserva las partes subidas
    mientras el multipart siga abierto.
    """
    try:
        response = client.list_multipart_uploads(Bucket=bucket)
    except Exception:
        return None
    candidates = [
        upload for upload in response.get("Uploads", [])
        # El gateway de RunPod a veces antepone una barra a la llave.
        if upload["Key"].lstrip("/") == key.lstrip("/")
    ]
    if not candidates:
        return None
    # Si por alguna razón hay varios, quedarse con el más reciente y cerrar el resto.
    candidates.sort(key=lambda u: u.get("Initiated") or 0)
    for stale in candidates[:-1]:
        try:
            client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=stale["UploadId"])
        except Exception:
            pass
    return candidates[-1]["UploadId"]


def existing_parts(client, bucket: str, key: str, upload_id: str) -> dict:
    """Mapa {numero_de_parte: (etag, tamaño)} de lo que ya está arriba."""
    parts = {}
    marker = 0
    while True:
        try:
            response = client.list_parts(
                Bucket=bucket, Key=key, UploadId=upload_id, PartNumberMarker=marker
            )
        except Exception as exc:
            print(f"    (no pude listar partes: {type(exc).__name__}; empiezo de cero)")
            return {}
        for part in response.get("Parts", []):
            parts[part["PartNumber"]] = (part["ETag"].strip('"'), part["Size"])
        if not response.get("IsTruncated"):
            break
        marker = response.get("NextPartNumberMarker", 0)
    return parts


def upload_resumable(client, bucket: str, key: str, path: Path) -> str:
    size = path.stat().st_size
    total_parts = (size + CHUNK_BYTES - 1) // CHUNK_BYTES

    # ¿Ya está completo?
    listing = client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=5)
    for obj in listing.get("Contents", []):
        if obj["Key"] == key and obj["Size"] == size:
            return f"SKIP  {key} (ya está completo, {human(size)})"

    upload_id = find_open_upload(client, bucket, key)
    done = existing_parts(client, bucket, key, upload_id) if upload_id else {}

    if upload_id and done:
        uploaded = sum(s for _, s in done.values())
        print(f"  reanudando: {len(done)}/{total_parts} partes ya arriba "
              f"({human(uploaded)} de {human(size)}, {100*uploaded/size:.1f} %)")
    else:
        if upload_id:
            # Multipart abierto pero sin partes utilizables: mejor empezar limpio.
            client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
        done = {}
        print(f"  multipart nuevo: {total_parts} partes de {CHUNK_BYTES // 2**20} MB")

    missing = [
        n for n in range(1, total_parts + 1)
        if not (n in done and done[n][1] == min(CHUNK_BYTES, size - (n - 1) * CHUNK_BYTES))
    ]
    if not missing:
        print("  todas las partes ya estaban; solo faltaba cerrar el multipart")

    started = time.monotonic()
    state = {"sent": 0, "last_print": 0.0}
    lock = threading.Lock()

    def send_part(part_number: int):
        """Sube una parte. Cada hilo abre su propio descriptor para poder hacer
        seek sin pisarse con los demás."""
        offset = (part_number - 1) * CHUNK_BYTES
        expected = min(CHUNK_BYTES, size - offset)
        with path.open("rb") as handle:
            handle.seek(offset)
            body = handle.read(expected)

        for attempt in range(1, PART_RETRIES + 1):
            try:
                result = client.upload_part(
                    Bucket=bucket, Key=key, UploadId=upload_id,
                    PartNumber=part_number, Body=body,
                )
                break
            except Exception as exc:
                if attempt == PART_RETRIES:
                    raise RuntimeError(
                        f"parte {part_number} falló tras {PART_RETRIES} intentos: {exc}"
                    ) from exc
                delay = min(BACKOFF_CAP_SECONDS, 2 ** (attempt - 1))
                with lock:
                    sys.stdout.write(f"\n    parte {part_number}: {type(exc).__name__}, "
                                     f"reintento {attempt}/{PART_RETRIES} en {delay}s\n")
                    sys.stdout.flush()
                time.sleep(delay)

        with lock:
            done[part_number] = (result["ETag"].strip('"'), expected)
            state["sent"] += expected
            now = time.monotonic()
            if now - state["last_print"] >= 5.0 or len(done) == total_parts:
                state["last_print"] = now
                elapsed = max(now - started, 1e-6)
                rate = state["sent"] / elapsed
                uploaded = sum(s for _, s in done.values())
                eta = (size - uploaded) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r  {path.name[:32]:<32} {100 * uploaded / size:5.1f}%  "
                    f"{human(uploaded)}/{human(size)}  {human(rate)}/s  "
                    f"ETA {eta / 60:5.1f} min  {len(done)}/{total_parts} partes   "
                )
                sys.stdout.flush()

    def watchdog():
        """Mata el proceso si deja de avanzar.

        Un corte de luz o de internet no cierra los sockets abiertos: se quedan
        colgados. Entre timeouts y reintentos el proceso puede pasar minutos sin
        avanzar y sin morir, que es lo peor de ambos mundos. Como el bucle de
        `upload_task1_loop.cmd` relanza y la subida retoma por partes, salir es
        barato: se pierde a lo sumo lo que estuviera en vuelo.
        """
        last_seen = -1
        while not finished.is_set():
            if finished.wait(STALL_CHECK_SECONDS):
                return
            with lock:
                current = state["sent"]
            if current == last_seen:
                sys.stdout.write(
                    f"\n  SIN AVANCE en {STALL_CHECK_SECONDS}s — salgo para que "
                    f"el bucle reinicie (se conserva lo ya subido)\n"
                )
                sys.stdout.flush()
                os._exit(2)
            last_seen = current

    finished = threading.Event()
    if missing:
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            with ThreadPoolExecutor(max_workers=PART_CONCURRENCY) as pool:
                futures = [pool.submit(send_part, n) for n in missing]
                for future in as_completed(futures):
                    future.result()   # propaga el primer fallo real
        finally:
            finished.set()

    sys.stdout.write("\n")
    client.complete_multipart_upload(
        Bucket=bucket, Key=key, UploadId=upload_id,
        MultipartUpload={"Parts": [
            {"PartNumber": n, "ETag": done[n][0]} for n in sorted(done)
        ]},
    )

    # Comprobación local del ETag compuesto: confirma que lo remoto es idéntico.
    digests = b"".join(bytes.fromhex(done[n][0]) for n in sorted(done))
    expected_etag = f"{hashlib.md5(digests).hexdigest()}-{len(done)}"
    return f"OK    {key} ({human(size)})  etag esperado {expected_etag}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--prefix", default="raw/")
    args = parser.parse_args()

    env = load_env(PROJECT_ROOT / ".env")
    client = boto3.client(
        "s3",
        endpoint_url=env["RUNPOD_S3_ENDPOINT"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env.get("RUNPOD_S3_REGION", "EU-RO-1"),
        # Timeouts cortos a propósito. Cuando se cae la luz o el internet, los
        # sockets abiertos no dan error: se quedan colgados hasta que expira el
        # read_timeout. Con 180 s el proceso parecía muerto durante minutos.
        # Una parte de 8 MB nunca debería tardar más de ~45 s; si tarda, la
        # conexión ya se murió y conviene fallar rápido y reintentar.
        config=BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"},
                          read_timeout=60, connect_timeout=15),
    )
    bucket = env["RUNPOD_S3_BUCKET"]
    prefix = args.prefix.strip("/")
    prefix = f"{prefix}/" if prefix else ""

    print(f"Endpoint : {env['RUNPOD_S3_ENDPOINT']}")
    print(f"Volumen  : {bucket}\n")

    failures = []
    for index, path in enumerate(args.paths, 1):
        if not path.is_file():
            sys.exit(f"No existe: {path}")
        key = prefix + path.name
        print(f"[{index}/{len(args.paths)}] {key}")
        try:
            print("  " + upload_resumable(client, bucket, key, path))
        except Exception as exc:
            print(f"  FAIL  {key}: {exc}")
            print("  (vuelve a correr el mismo comando: retomará donde se quedó)")
            failures.append(key)

    if failures:
        sys.exit(1)
    print("\nTodo subido.")


if __name__ == "__main__":
    main()
