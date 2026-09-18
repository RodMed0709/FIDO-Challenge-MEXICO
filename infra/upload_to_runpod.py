#!/usr/bin/env python3
"""Upload local dataset files to the RunPod S3-backed network volume.

Reads credentials from the project .env. Designed for the FIDO dataset, which
is tens of thousands of small PNGs: uploading those one-by-one over HTTP is
pathologically slow, so directories are packed into tar archives by default and
unpacked inside the pod.

Usage
-----
    # Upload the zips Google Drive gave you, as-is
    python infra/upload_to_runpod.py data/*.zip --prefix raw/

    # Pack a directory tree into one tar and upload it
    python infra/upload_to_runpod.py "data/Task 2" --tar --prefix raw/

    # Upload a tree file-by-file (only sensible for a few large files)
    python infra/upload_to_runpod.py data/weights --no-tar --prefix weights/

    # See what would happen without transferring anything
    python infra/upload_to_runpod.py data/ --tar --dry-run

    # List what is already on the volume
    python infra/upload_to_runpod.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    sys.exit("boto3 no instalado. Corre:  pip install boto3")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# RunPod fronts its S3 gateway with Cloudflare, which kills any single request
# that takes much over ~100 s with a 524. For a multipart upload the time one
# part takes is (chunk_size * concurrent_streams) / available_bandwidth, so on a
# slow uplink big chunks are actively harmful: 64 MB parts over 8 streams at
# ~700 KB/s takes ~12 minutes per part and 524s every time.
#
# Keep chunk_size * streams small enough that a part finishes well inside the
# proxy's patience. 8 MB * 4 streams = 32 MB in flight, ~45 s at 700 KB/s.
MULTIPART_CHUNK_BYTES = 8 * 1024 * 1024
MULTIPART_THRESHOLD_BYTES = 8 * 1024 * 1024
STREAMS_PER_FILE = 4

# Sequential by design. On a bandwidth-limited uplink, uploading files
# concurrently splits the same pipe more ways, making every individual part
# slower and 524s more likely, for no throughput gain.
PARALLEL_FILES = 1

# A 28 GB transfer runs for hours; a single transient proxy hiccup must not end
# it. Retried per file, since boto3 cannot resume a failed multipart upload.
MAX_FILE_ATTEMPTS = 6


def load_env(path: Path) -> dict:
    """Parse a KEY=VALUE .env file. Ignores blanks, comments and inline quotes."""
    if not path.exists():
        sys.exit(f"No encuentro {path}. Ese archivo trae las credenciales de RunPod.")
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def make_client(env: dict):
    missing = [
        key
        for key in ("RUNPOD_S3_ENDPOINT", "RUNPOD_S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not env.get(key)
    ]
    if missing:
        sys.exit(f"Faltan variables en .env: {', '.join(missing)}")

    return boto3.client(
        "s3",
        endpoint_url=env["RUNPOD_S3_ENDPOINT"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env.get("RUNPOD_S3_REGION", "EU-RO-1"),
        # Retries matter here: multi-GB uploads over a home connection will hit
        # transient resets, and losing a 60 GB transfer to one blip is brutal.
        config=BotoConfig(
            retries={"max_attempts": 10, "mode": "adaptive"},
            read_timeout=300,
            connect_timeout=60,
        ),
    )


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


class Progress:
    """Thread-safe byte counter that prints a single updating line."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.seen = 0
        self.started = time.monotonic()
        self._last_print = 0.0
        self._lock = threading.Lock()

    def __call__(self, chunk: int):
        with self._lock:
            self.seen += chunk
            now = time.monotonic()
            # Throttle: a 28 GB transfer would otherwise emit tens of thousands
            # of lines into the background-task log.
            if now - self._last_print < 5.0 and self.seen < self.total:
                return
            self._last_print = now
            elapsed = max(now - self.started, 1e-6)
            rate = self.seen / elapsed
            pct = (self.seen / self.total * 100) if self.total else 0.0
            eta = (self.total - self.seen) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  {self.label[:48]:<48} {pct:5.1f}%  "
                f"{human(self.seen)}/{human(self.total)}  "
                f"{human(rate)}/s  ETA {eta/60:.1f} min   "
            )
            sys.stdout.flush()

    def done(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def remote_size(client, bucket: str, key: str):
    """Return the size of an existing object, or None if it is not there.

    Uses ListObjectsV2 rather than HeadObject on purpose: RunPod's S3 gateway
    answers HeadObject on a missing key with 403 Forbidden instead of the 404
    the API contract calls for, which is indistinguishable from a real
    permissions failure. Listing by exact prefix sidesteps that entirely.
    """
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=10)
    except ClientError:
        # If we cannot check, assume absent and let the upload itself surface
        # any real error.
        return None
    for obj in response.get("Contents", []):
        if obj["Key"] == key:
            return obj["Size"]
    return None


def upload_one(client, bucket: str, local: Path, key: str, *, force: bool,
               quiet: bool, chunk_bytes: int = MULTIPART_CHUNK_BYTES,
               streams: int = STREAMS_PER_FILE) -> str:
    size = local.stat().st_size
    existing = remote_size(client, bucket, key)
    if existing == size and not force:
        return f"  SKIP  {key}  (ya está, {human(size)})"

    config = TransferConfig(
        multipart_threshold=min(chunk_bytes, MULTIPART_THRESHOLD_BYTES),
        multipart_chunksize=chunk_bytes,
        max_concurrency=streams,
        use_threads=True,
    )

    last_error = None
    for attempt in range(1, MAX_FILE_ATTEMPTS + 1):
        progress = Progress(local.name, size) if not quiet else None
        try:
            client.upload_file(str(local), bucket, key, Config=config, Callback=progress)
            if progress:
                progress.done()
            suffix = f"  (intento {attempt})" if attempt > 1 else ""
            return f"  OK    {key}  ({human(size)}){suffix}"
        except Exception as exc:
            if progress:
                progress.done()
            last_error = exc
            # Abandoned multipart uploads keep consuming volume space, so clear
            # them before trying again.
            try:
                for upload in client.list_multipart_uploads(
                    Bucket=bucket, Prefix=key
                ).get("Uploads", []):
                    if upload["Key"] == key:
                        client.abort_multipart_upload(
                            Bucket=bucket, Key=key, UploadId=upload["UploadId"]
                        )
            except Exception:
                pass

            if attempt == MAX_FILE_ATTEMPTS:
                break
            backoff = min(60, 5 * 2 ** (attempt - 1))
            print(f"  RETRY {key}  intento {attempt}/{MAX_FILE_ATTEMPTS} falló "
                  f"({type(exc).__name__}); reintento en {backoff}s")
            time.sleep(backoff)

    return f"  FAIL  {key}: {last_error}"


def make_tar(source: Path, staging: Path) -> Path:
    """Pack a directory into an uncompressed tar in `staging`.

    Uncompressed on purpose: the payload is PNGs, which are already compressed,
    so gzip would burn CPU for roughly nothing and slow the upload down.
    """
    archive = staging / f"{source.name.replace(' ', '_')}.tar"
    files = [p for p in source.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"  empacando {len(files):,} archivos ({human(total)}) -> {archive.name}")

    with tarfile.open(archive, "w") as tar:
        for index, path in enumerate(files, 1):
            tar.add(path, arcname=str(path.relative_to(source.parent)))
            if index % 2000 == 0:
                sys.stdout.write(f"\r    {index:,}/{len(files):,} archivos")
                sys.stdout.flush()
    sys.stdout.write("\r" + " " * 60 + "\r")
    print(f"  tar listo: {human(archive.stat().st_size)}")
    return archive


def cmd_list(client, bucket: str):
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    total = 0
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            print(f"  {obj['Size']:>14,}  {obj['Key']}")
            count += 1
            total += obj["Size"]
    if count == 0:
        print("  (volumen vacío)")
    else:
        print(f"\n  {count} objetos, {human(total)} en total")


def main():
    parser = argparse.ArgumentParser(
        description="Sube archivos al volumen S3 de RunPod.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Archivos o carpetas a subir")
    parser.add_argument("--prefix", default="raw/", help="Prefijo de destino en el bucket (default: raw/)")
    parser.add_argument("--tar", dest="tar", action="store_true", default=None,
                        help="Empaquetar carpetas en un tar antes de subir (default para carpetas)")
    parser.add_argument("--no-tar", dest="tar", action="store_false",
                        help="Subir las carpetas archivo por archivo")
    parser.add_argument("--staging", type=Path, default=None,
                        help="Dónde escribir los tars temporales (default: temp del sistema)")
    parser.add_argument("--streams", type=int, default=STREAMS_PER_FILE,
                        help="Streams concurrentes por archivo. Bájalo si el enlace es lento "
                             "(cada parte tarda chunk x streams / ancho de banda)")
    parser.add_argument("--chunk-mb", type=int, default=MULTIPART_CHUNK_BYTES // (1024 * 1024),
                        help="Tamaño de cada parte multipart en MB. Bájalo si salen errores 524 "
                             "(el proxy corta las peticiones lentas); súbelo si tu enlace es rápido")
    parser.add_argument("--force", action="store_true", help="Resubir aunque el objeto ya exista con el mismo tamaño")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar el plan sin transferir nada")
    parser.add_argument("--list", action="store_true", help="Listar el contenido del volumen y salir")
    args = parser.parse_args()

    env = load_env(ENV_PATH)
    client = make_client(env)
    bucket = env["RUNPOD_S3_BUCKET"]

    print(f"Endpoint : {env['RUNPOD_S3_ENDPOINT']}")
    print(f"Volumen  : {bucket}\n")

    if args.list:
        cmd_list(client, bucket)
        return

    if not args.paths:
        parser.error("Dame al menos una ruta que subir, o usa --list")

    prefix = args.prefix.strip("/")
    prefix = f"{prefix}/" if prefix else ""

    # Resolve every input into concrete (local_file, remote_key) pairs first, so
    # the plan can be printed and sanity-checked before a byte moves.
    staging_ctx = None
    staging_dir = args.staging
    if staging_dir is None:
        staging_ctx = tempfile.TemporaryDirectory(prefix="fido_upload_")
        staging_dir = Path(staging_ctx.name)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        jobs: list[tuple[Path, str]] = []
        for path in args.paths:
            if not path.exists():
                sys.exit(f"No existe: {path}")

            if path.is_file():
                jobs.append((path, prefix + path.name))
                continue

            use_tar = True if args.tar is None else args.tar
            if use_tar:
                if args.dry_run:
                    files = [p for p in path.rglob("*") if p.is_file()]
                    total = sum(p.stat().st_size for p in files)
                    print(f"  [dry-run] empacaría {path} -> "
                          f"{prefix}{path.name.replace(' ', '_')}.tar "
                          f"({len(files):,} archivos, {human(total)})")
                    continue
                archive = make_tar(path, staging_dir)
                jobs.append((archive, prefix + archive.name))
            else:
                for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
                    rel = file_path.relative_to(path.parent).as_posix()
                    jobs.append((file_path, prefix + rel))

        if args.dry_run:
            total = sum(local.stat().st_size for local, _ in jobs)
            for local, key in jobs:
                print(f"  [dry-run] {local}  ->  s3://{bucket}/{key}  ({human(local.stat().st_size)})")
            if jobs:
                print(f"\n  Total: {len(jobs)} objetos, {human(total)}")
            return

        if not jobs:
            print("  Nada que subir.")
            return

        total = sum(local.stat().st_size for local, _ in jobs)
        print(f"Subiendo {len(jobs)} objeto(s), {human(total)} en total\n")
        started = time.monotonic()

        chunk_bytes = args.chunk_mb * 1024 * 1024
        failures = []

        if PARALLEL_FILES <= 1:
            for index, (local, key) in enumerate(jobs, 1):
                print(f"[{index}/{len(jobs)}] {key}")
                message = upload_one(client, bucket, local, key, force=args.force,
                                     quiet=False, chunk_bytes=chunk_bytes,
                                     streams=args.streams)
                print(message)
                if message.lstrip().startswith("FAIL"):
                    failures.append(key)
        else:
            with ThreadPoolExecutor(max_workers=PARALLEL_FILES) as pool:
                futures = {
                    pool.submit(upload_one, client, bucket, local, key,
                                force=args.force, quiet=True,
                                chunk_bytes=chunk_bytes, streams=args.streams): key
                    for local, key in jobs
                }
                for done_count, future in enumerate(as_completed(futures), 1):
                    try:
                        message = future.result()
                    except Exception as exc:
                        message = f"  FAIL  {futures[future]}: {exc}"
                    if message.lstrip().startswith("FAIL"):
                        failures.append(futures[future])
                    print(f"[{done_count}/{len(jobs)}]{message}")

        elapsed = time.monotonic() - started
        print(f"\nListo en {elapsed/60:.1f} min  ({human(total/max(elapsed, 1e-6))}/s promedio)")
        if failures:
            print(f"\n{len(failures)} objeto(s) fallaron:")
            for key in failures:
                print(f"  - {key}")
            print("Vuelve a correr el mismo comando: los completos se saltan.")
            sys.exit(1)

    finally:
        if staging_ctx is not None:
            staging_ctx.cleanup()


if __name__ == "__main__":
    main()
