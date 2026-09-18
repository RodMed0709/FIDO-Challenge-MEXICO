# Desmantelamiento del volumen de RunPod — rescate del 2026-09-18

El network volume `FIDO` (600 GB, RunPod, datacenter EU-RO-1, id `<VOLUME_ID>`) se
da de baja por costo. Este documento registra qué había, qué se bajó, qué se
descartó y con qué evidencia — para que el proyecto se pueda retomar sin el
volumen.

**Después de este rescate el volumen se borra. Nada de lo que quedó allá es
recuperable.**

## Dónde quedó cada cosa

| Qué | Dónde vive ahora |
|---|---|
| Código, docs, ledger, `ATTACK_LADDER.md`, `RESULTS.md` | este repo, sincronizado con `origin/master` en GitHub |
| Dataset completo (20 escenarios, Task 1 + Task 2) | `data/` local, en zips originales |
| Submissions `r00`…`r06` con sus pesos | `submissions/` local |
| Checkpoints de entrenamiento del pod | `pod_rescue/FIDO_CHALLENGE/checkpoints/` |
| Logs de corridas del pod | `pod_rescue/*.log` |
| Caché de casos de Task 1 | `pod_rescue/cache/task1_cases.json` |
| Mock Test (tar original) | `pod_rescue/raw/Mock_Test.tar` |

`pod_rescue/` está en `.gitignore`: contiene pesos y una copia de `.env` con
credenciales. **No se sube a GitHub.** Es un respaldo local; si el disco D: se
pierde, se pierde.

## Qué se bajó

1719 objetos, 1247 MiB, 0 errores. Inventario completo en
`pod_rescue/MANIFEST.csv`.

| Prefijo del volumen | Contenido | Tamaño |
|---|---|---|
| `FIDO_CHALLENGE/` | copia (vieja) del repo + `checkpoints/` + `experiments/` + Mock Test extraído | 1053 MiB |
| `checkpoints_backup/` | `model_0_auc0.5681_ep2.pth`, `model_1_ep5_err203px.pth` | 67 MiB |
| `cache/task1_cases.json` | índice de casos de Task 1 | 3 MiB |
| raíz del volumen | 30+ logs de corridas y scripts de orquestación | 6 MiB |
| `raw/Mock_Test.tar` | Mock Test sin extraer | 119 MiB |

Se excluyeron `__pycache__/` y `.pytest_cache/`.

### Checkpoints rescatados (24 pesos)

`smoke_keypoint`, `smoke_task1_unet_fold1`, `smoke_task1_weighted`,
`smoke_task2`, `smoke_task2_crosscorr{,2,3,4}`, `smoke_task2_v{2,3,4}`,
`task1_clean_cnn`, `task1_dist_v2`, `task1_keypoint`, `task1_kp_dinov2`,
`task1_real`, `task1_resnet18_fpn`, `task2_common_cnn`, `task2_dinov2`,
`task2_fixed`, `task2_real`, `task2_scale_aug`, más los dos de
`checkpoints_backup/`.

### Archivos que solo existían en el volumen

Tres archivos de experimento no estaban en el repo. Se copiaron a `experiments/`
para que viajen a GitHub:

- `experiments/70-t2-oracle/CONVENTION.md` — barrido de las 8 convenciones
  en-face sobre n=1214 casos, con null de 200 muestras por keypoint.
- `experiments/70-t2-oracle/ORACLE_FIXED.md` — medición de señal del oráculo de
  Task 2 sobre 120 casos con la convención `transpose__flip_u__flip_v`.
- `experiments/81-t2-scale-augmentation/enface_cache.log`.

También se rescató `src/fido/test_geometry.py` (versión vieja, antes de moverse
a `src/fido/tests/`) dentro de `pod_rescue/`.

## Qué se descartó, y por qué es seguro

| Prefijo | Tamaño | Razón |
|---|---|---|
| `raw/Task1/*.zip`, `raw/Task2/*.zip` | ~89 GB | copia idéntica en `data/` local |
| `data/Task1/`, `data/Task2/` (extraído) | grande | derivable de los zips |
| `venv/` | grande | reproducible con `infra/bootstrap_pod.sh` |
| `test/`, `.s3compat_uploads/` | 0 | vacíos |

**Verificación de los zips antes de descartarlos** — los 20 pasaron dos pruebas:

1. Tamaño exacto en bytes, remoto contra local, 20/20.
2. Comparación binaria por SHA-256 de tres ventanas de 4 MiB por archivo
   (primeros 4 MiB, 4 MiB del centro, últimos 4 MiB), remoto contra local.
   20/20 idénticos, 0 discrepancias.

No se hizo hash completo de los 89 GB. La combinación de tamaño exacto más las
tres ventanas descarta truncamiento y corrupción parcial, que son los modos de
fallo reales de una subida por red.

## Lo que NO se pudo rescatar

`infra/precompute_task2_enface.py` escribía su caché en `/root/data_cache/Task2_enface`,
que es disco efímero del pod, **no** el volumen. Esa caché ya no existía al
momento del rescate. Se regenera corriendo el script otra vez sobre los datos.

## Cómo retomar el proyecto sin el volumen

1. `git clone` de este repo.
2. Descomprimir `data/Task 1/*.zip` y `data/Task 2/*.zip` donde vaya a correr el
   entrenamiento.
3. Crear volumen y pod nuevos con `infra/create_pod.py`, subir datos con
   `infra/resumable_upload.py`, montar entorno con `infra/bootstrap_pod.sh`
   (exige PyTorch ≥ 2.7 + CUDA 12.8 para la RTX 5090 / `sm_120`).
4. Copiar los checkpoints de `pod_rescue/FIDO_CHALLENGE/checkpoints/` al volumen
   nuevo si se quiere continuar desde un peso ya entrenado.
5. Regenerar la caché en-face de Task 2 con `infra/precompute_task2_enface.py`.

El estado científico del proyecto — qué se probó, qué falló y por qué — está en
`ATTACK_LADDER.md`, `RESULTS.md` y `ledger/`. Ese es el activo real; los pesos
son reproducibles, el razonamiento no.
