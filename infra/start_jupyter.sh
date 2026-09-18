#!/usr/bin/env bash
# Start JupyterLab on a RunPod pod, served from the network volume.
#
#     bash infra/start_jupyter.sh
#
# Expose port 8888 on the pod (RunPod: "HTTP Ports" in the pod config) and open
# the proxied URL RunPod gives you: https://<POD_ID>-8888.proxy.runpod.net

set -euo pipefail

VOLUME="${VOLUME:-/workspace}"
VENV="$VOLUME/venv"
PORT="${PORT:-8888}"

if [ ! -d "$VENV" ]; then
  echo "ERROR: no hay venv en $VENV. Corre primero: bash infra/bootstrap_pod.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# The pod is already behind RunPod's authenticated proxy, so a second token
# prompt only gets in the way. If you expose the port publicly by other means,
# drop --NotebookApp.token='' and use a real token.
exec jupyter lab \
  --ip=0.0.0.0 \
  --port="$PORT" \
  --no-browser \
  --allow-root \
  --notebook-dir="$VOLUME" \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.disable_check_xsrf=True
