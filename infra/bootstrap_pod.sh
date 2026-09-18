#!/usr/bin/env bash
# Prepare a fresh RunPod pod for FIDO training.
#
# Run once per pod:
#     bash infra/bootstrap_pod.sh
#
# Assumes the network volume is mounted at $VOLUME (default /workspace) and that
# the dataset zips were already pushed there with infra/upload_to_runpod.py.

set -euo pipefail

VOLUME="${VOLUME:-/workspace}"
RAW_DIR="$VOLUME/raw"          # where upload_to_runpod.py puts the zips
DATA_DIR="$VOLUME/data"        # where the extracted dataset lands
VENV="$VOLUME/venv"

echo "=== FIDO pod bootstrap ==="
echo "volumen : $VOLUME"

if [ ! -d "$VOLUME" ]; then
  echo "ERROR: $VOLUME no existe. ¿Montaste el network volume en el pod?" >&2
  exit 1
fi

# --- GPU check -------------------------------------------------------------
# The RTX 5090 is Blackwell (sm_120). PyTorch wheels built for cu121/cu124 ship
# no kernels for that arch, so they import fine and then fail at the first CUDA
# op. Pinning the cu128 index is the whole reason this script exists.
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "ERROR: nvidia-smi falló. El pod no ve GPU." >&2
  exit 1
}

# --- system deps -----------------------------------------------------------
echo
echo "--- paquetes de sistema ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq unzip pigz git curl libgl1 libglib2.0-0 >/dev/null
echo "ok"

# --- python env ------------------------------------------------------------
echo
echo "--- entorno python ---"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  echo "venv creado en $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

echo "instalando torch cu128 (obligatorio para sm_120 / RTX 5090)..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -q \
  numpy opencv-python-headless pillow scipy scikit-image \
  tqdm pyyaml einops timm kornia \
  matplotlib pandas \
  boto3 pytest

echo "instalando JupyterLab..."
pip install -q jupyterlab ipywidgets ipykernel
# Register the volume venv as a named kernel so notebooks opened from any
# Jupyter instance on the pod can select it.
python -m ipykernel install --sys-prefix --name fido --display-name "FIDO (cu128)"

python - <<'PY'
import torch
print(f"torch      : {torch.__version__}")
print(f"cuda build : {torch.version.cuda}")
print(f"cuda avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device     : {name}  sm_{cap[0]}{cap[1]}")
    arch_list = torch.cuda.get_arch_list()
    print(f"arch list  : {arch_list}")
    if f"sm_{cap[0]}{cap[1]}" not in arch_list:
        raise SystemExit(
            f"FATAL: esta build de torch no trae kernels para sm_{cap[0]}{cap[1]}. "
            "Reinstala desde el index cu128."
        )
    # Prove a real kernel launches, not just that the runtime loaded.
    x = torch.randn(2048, 2048, device="cuda")
    assert torch.isfinite(x @ x).all()
    print("smoke test : OK")
PY

# --- dataset ---------------------------------------------------------------
echo
echo "--- dataset ---"
mkdir -p "$DATA_DIR"

shopt -s nullglob globstar
zips=("$RAW_DIR"/*.zip "$RAW_DIR"/**/*.zip)
if [ ${#zips[@]} -eq 0 ]; then
  echo "AVISO: no hay zips en $RAW_DIR. Súbelos con infra/upload_to_runpod.py."
else
  for zip_path in "${zips[@]}"; do
    marker="$DATA_DIR/.extracted_$(basename "$zip_path" .zip)"
    if [ -f "$marker" ]; then
      echo "  ya extraído: $(basename "$zip_path")"
      continue
    fi
    echo "  extrayendo $(basename "$zip_path")..."
    unzip -q -o "$zip_path" -d "$DATA_DIR"
    touch "$marker"
  done
  echo
  echo "  contenido de $DATA_DIR:"
  ls -1 "$DATA_DIR" | head -20
  echo "  archivos: $(find "$DATA_DIR" -type f | wc -l)"
  echo "  tamaño  : $(du -sh "$DATA_DIR" | cut -f1)"
fi

echo
echo "=== bootstrap terminado ==="
echo "activa el entorno con:  source $VENV/bin/activate"
echo "levanta jupyter con  :  bash infra/start_jupyter.sh"
