#!/bin/bash
# Copia a disco LOCAL del pod solo los archivos de Task 1 que el entrenamiento
# de segmentacion (T1-R5) lee de verdad.
#
# Por que: /workspace es MooseFS (FUSE sobre red). Medido el 2026-08-17:
# ~27 ms por lectura aleatoria de archivo, y el entrenamiento hace 4 lecturas
# por caso sobre 49k casos = ~11 min por epoca, con la GPU al 0% esperando I/O.
# El disco local del contenedor escribe a 4.3 GB/s y tiene 59 GB libres; los
# B-scans + segmentaciones de las 10 escenas pesan ~22 GB. Se paga una copia
# UNA VEZ y todas las epocas siguientes leen a velocidad de NVMe.
#
# NO se copia `Stereo Left/*/microscope.png` (el fundus, 43 GB): T1-R5 no lo usa.
# El entrenamiento de keypoint (T1-R3) si lo necesita y se cachea aparte, porque
# ambos juntos (65 GB) no caben en los 59 GB del overlay.
#
# Se paraleliza por escenario (10 tar simultaneos) y no por archivo: `tar` lee
# secuencialmente, que sobre MooseFS es ~2x mas rapido por archivo que el acceso
# aleatorio (14 ms vs 27 ms), y evita crear 247k procesos de `cp`.
set -uo pipefail

SRC=/workspace/data/Task1
DST=/root/data_cache/Task1

echo "=== COPIA LOCAL TASK1 INICIADA $(date) ==="
mkdir -p "$DST"

copy_scenario() {
  local scen="$1"
  local name
  name=$(basename "$scen")
  mkdir -p "$DST/$name"
  # Numerical (Ground Truth por frame) + Bscan (2 imagenes + 2 mascaras por frame).
  # Un solo tar por escenario: una pasada secuencial sobre el arbol remoto.
  ( cd "$scen" && tar cf - "Numerical" "iOCT Microscope/Bscan" 2>/dev/null ) \
    | ( cd "$DST/$name" && tar xf - 2>/dev/null )
  echo "[$(date +%H:%M:%S)] listo $name"
}
export -f copy_scenario
export DST

ls -d "$SRC"/Scenario_* | xargs -P 10 -I{} bash -c 'copy_scenario "$@"' _ {}

echo "=== COPIA LOCAL TASK1 TERMINADA $(date) ==="
du -sh "$DST" 2>/dev/null
df -h / | tail -1
