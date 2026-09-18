# Baseline de correspondencia Task 2 — T2-R10

## Corrida

- Comando exacto: `python analysis/measure_task2_correspondence.py --root /workspace/data/Task2 --checkpoint /workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth --output experiments/80-t2-diagnostics/correspondence.md --jsonl experiments/80-t2-diagnostics/correspondence_cases.jsonl --n-folds 5 --fold 0 --k 64 --seed 0 --device cuda --oct-shuffle`
- Checkpoint: `/workspace/FIDO_CHALLENGE/checkpoints/task2_fixed/model_1.pth`
- Dispositivo: `cuda`
- Raíz efectiva: `/workspace/data/Task2`
- Casos medidos: **248** de 10 escenario(s)
- Split: GroupKFold exacto: fold 0/5.
- Semilla: 0; K=64 puntos por caso

## Resultados

| Métrica | Definición | Resultado |
|---|---|---:|
| M1, positivo vecino | Positivo entre el top-1% (41/4096) de todas las celdas fundus; rank mediano como percentil | recall **0.34%**; percentil **81.836%** |
| M1, positivo bilineal | Misma consulta, score del feature fundus bilineal en la coordenada GT | recall **0.39%**; percentil **82.446%** |
| M2 | Pico duro del mapa de correlación exacto del modelo, distancia al centro GT, con cabeza de regresión omitida | ≤10 px **6.9%**; distancia mediana **37.12 px** |
| M3 | Masa softmax del pico y correlación de Pearson punto-biserial con `M2 ≤10 px` | masa mediana **0.090954**; correlación **-0.017** |

La correlación del modelo no está condicionada por la rotación ni escala que produce su cabeza;
por eso “dar θ/s GT” equivale aquí a omitir completamente esa cabeza y evaluar el pico contra el
centro calculado directamente con la matriz GT. No se modificó el mapa de correlación.

## ¿Reproduce 28.6%?

**NO. El baseline medido es 6.9% frente al 28.6% citado.**

La comparación no es equivalente al fold real de entrenamiento: el disco local solo contiene cinco
casos de Mock Test y un único escenario, de modo que `GroupKFold(5)` es imposible. El 28.6% y el
6.9% deben conservarse ambos; este último es la referencia local reproducible.

## M3 por caso

| Caso | Distancia M2 (px) | Masa del pico | Correcto ≤10 px |
|---|---:|---:|---:|
| `Scenario_09/00000` | 23.23 | 0.121071 | no |
| `Scenario_09/00001` | 36.30 | 0.178139 | no |
| `Scenario_09/00002` | 61.16 | 0.133897 | no |
| `Scenario_09/00003` | 9.05 | 0.075833 | sí |
| `Scenario_09/00004` | 30.86 | 0.103928 | no |
| `Scenario_09/00005` | 19.44 | 0.262911 | no |
| `Scenario_09/00006` | 16.25 | 0.052868 | no |
| `Scenario_09/00007` | 31.46 | 0.062069 | no |
| `Scenario_09/00008` | 72.39 | 0.058109 | no |
| `Scenario_09/00009` | 12.36 | 0.247975 | no |
| `Scenario_09/00010` | 10.40 | 0.116359 | no |
| `Scenario_09/00011` | 96.39 | 0.134691 | no |
| `Scenario_09/00012` | 19.23 | 0.175162 | no |
| `Scenario_09/00013` | 73.99 | 0.365546 | no |
| `Scenario_09/00014` | 37.60 | 0.126423 | no |
| `Scenario_09/00015` | 21.23 | 0.129456 | no |
| `Scenario_09/00016` | 26.53 | 0.162988 | no |
| `Scenario_09/00017` | 50.74 | 0.078055 | no |
| `Scenario_09/00018` | 5.98 | 0.168857 | sí |
| `Scenario_09/00019` | 11.52 | 0.184696 | no |
| `Scenario_09/00020` | 11.14 | 0.140278 | no |
| `Scenario_09/00021` | 39.75 | 0.071790 | no |
| `Scenario_09/00022` | 27.15 | 0.230958 | no |
| `Scenario_09/00023` | 77.34 | 0.240692 | no |
| `Scenario_09/00024` | 40.95 | 0.102683 | no |
| `Scenario_09/00025` | 56.18 | 0.081400 | no |
| `Scenario_09/00026` | 59.46 | 0.109971 | no |
| `Scenario_09/00027` | 28.15 | 0.189600 | no |
| `Scenario_09/00028` | 30.54 | 0.248145 | no |
| `Scenario_09/00029` | 65.00 | 0.146734 | no |
| `Scenario_09/00030` | 18.79 | 0.525050 | no |
| `Scenario_09/00031` | 57.77 | 0.508040 | no |
| `Scenario_09/00032` | 13.93 | 0.070602 | no |
| `Scenario_09/00033` | 9.85 | 0.114645 | sí |
| `Scenario_09/00034` | 36.84 | 0.152750 | no |
| `Scenario_09/00035` | 58.06 | 0.181214 | no |
| `Scenario_09/00036` | 65.07 | 0.137293 | no |
| `Scenario_09/00037` | 79.98 | 0.077182 | no |
| `Scenario_09/00038` | 94.60 | 0.305358 | no |
| `Scenario_09/00039` | 76.91 | 0.434459 | no |
| `Scenario_09/00040` | 27.22 | 0.047468 | no |
| `Scenario_09/00041` | 16.06 | 0.108833 | no |
| `Scenario_09/00042` | 23.08 | 0.105234 | no |
| `Scenario_09/00043` | 15.31 | 0.058341 | no |
| `Scenario_09/00044` | 39.12 | 0.099917 | no |
| `Scenario_09/00045` | 38.38 | 0.255207 | no |
| `Scenario_09/00046` | 54.19 | 0.369757 | no |
| `Scenario_09/00047` | 110.47 | 0.113968 | no |
| `Scenario_09/00048` | 96.69 | 0.323317 | no |
| `Scenario_09/00049` | 15.77 | 0.059247 | no |
| `Scenario_09/00050` | 41.97 | 0.086007 | no |
| `Scenario_09/00051` | 37.37 | 0.048607 | no |
| `Scenario_09/00052` | 10.09 | 0.111725 | no |
| `Scenario_09/00053` | 38.61 | 0.256026 | no |
| `Scenario_09/00054` | 26.58 | 0.196176 | no |
| `Scenario_09/00055` | 24.20 | 0.085123 | no |
| `Scenario_09/00056` | 27.05 | 0.264569 | no |
| `Scenario_09/00057` | 28.87 | 0.070811 | no |
| `Scenario_09/00058` | 32.78 | 0.042108 | no |
| `Scenario_09/00059` | 34.18 | 0.108091 | no |
| `Scenario_09/00060` | 27.58 | 0.089693 | no |
| `Scenario_09/00061` | 28.17 | 0.195365 | no |
| `Scenario_09/00062` | 5.70 | 0.046037 | sí |
| `Scenario_09/00063` | 20.66 | 0.164044 | no |
| `Scenario_09/00064` | 31.84 | 0.073331 | no |
| `Scenario_09/00065` | 74.01 | 0.180982 | no |
| `Scenario_09/00066` | 15.65 | 0.090781 | no |
| `Scenario_09/00067` | 43.70 | 0.106248 | no |
| `Scenario_09/00068` | 55.96 | 0.083482 | no |
| `Scenario_09/00069` | 13.07 | 0.200921 | no |
| `Scenario_09/00070` | 40.54 | 0.074117 | no |
| `Scenario_09/00071` | 25.49 | 0.118009 | no |
| `Scenario_09/00072` | 26.84 | 0.044943 | no |
| `Scenario_09/00073` | 3.12 | 0.111752 | sí |
| `Scenario_09/00074` | 26.76 | 0.212621 | no |
| `Scenario_09/00075` | 48.59 | 0.219499 | no |
| `Scenario_09/00076` | 24.07 | 0.121278 | no |
| `Scenario_09/00077` | 4.56 | 0.153360 | sí |
| `Scenario_09/00078` | 35.85 | 0.084955 | no |
| `Scenario_09/00079` | 41.67 | 0.172607 | no |
| `Scenario_09/00080` | 67.37 | 0.110772 | no |
| `Scenario_09/00081` | 25.67 | 0.080495 | no |
| `Scenario_09/00082` | 110.06 | 0.066012 | no |
| `Scenario_09/00083` | 15.40 | 0.149565 | no |
| `Scenario_09/00084` | 71.38 | 0.062792 | no |
| `Scenario_09/00085` | 50.02 | 0.207231 | no |
| `Scenario_09/00086` | 28.56 | 0.173344 | no |
| `Scenario_09/00087` | 33.07 | 0.126223 | no |
| `Scenario_09/00088` | 22.50 | 0.055666 | no |
| `Scenario_09/00089` | 20.48 | 0.119549 | no |
| `Scenario_09/00090` | 43.41 | 0.092230 | no |
| `Scenario_09/00091` | 24.61 | 0.122858 | no |
| `Scenario_09/00092` | 24.86 | 0.106676 | no |
| `Scenario_09/00093` | 66.07 | 0.157096 | no |
| `Scenario_09/00094` | 35.17 | 0.081432 | no |
| `Scenario_09/00095` | 12.50 | 0.185882 | no |
| `Scenario_09/00096` | 92.63 | 0.096665 | no |
| `Scenario_09/00097` | 75.68 | 0.036058 | no |
| `Scenario_09/00098` | 55.41 | 0.044403 | no |
| `Scenario_09/00099` | 23.23 | 0.083525 | no |
| `Scenario_09/00100` | 35.42 | 0.181692 | no |
| `Scenario_09/00101` | 11.86 | 0.122060 | no |
| `Scenario_09/00102` | 23.91 | 0.065218 | no |
| `Scenario_09/00103` | 42.26 | 0.053709 | no |
| `Scenario_09/00104` | 38.16 | 0.070266 | no |
| `Scenario_09/00105` | 52.98 | 0.026233 | no |
| `Scenario_09/00106` | 46.43 | 0.052651 | no |
| `Scenario_09/00107` | 16.22 | 0.055469 | no |
| `Scenario_09/00108` | 38.56 | 0.036364 | no |
| `Scenario_09/00109` | 41.32 | 0.091128 | no |
| `Scenario_09/00110` | 18.25 | 0.067609 | no |
| `Scenario_09/00111` | 6.42 | 0.174803 | sí |
| `Scenario_09/00112` | 46.03 | 0.056504 | no |
| `Scenario_09/00113` | 15.48 | 0.082502 | no |
| `Scenario_09/00114` | 15.63 | 0.111094 | no |
| `Scenario_09/00115` | 31.36 | 0.097663 | no |
| `Scenario_09/00116` | 6.27 | 0.099984 | sí |
| `Scenario_09/00117` | 7.46 | 0.095073 | sí |
| `Scenario_09/00118` | 9.40 | 0.079379 | sí |
| `Scenario_09/00119` | 53.40 | 0.405844 | no |
| `Scenario_09/00120` | 46.34 | 0.125974 | no |
| `Scenario_09/00121` | 35.45 | 0.118135 | no |
| `Scenario_09/00122` | 18.01 | 0.115950 | no |
| `Scenario_09/00123` | 110.80 | 0.016196 | no |
| `Scenario_09/00124` | 163.43 | 0.032369 | no |
| `Scenario_09/00125` | 41.41 | 0.015873 | no |
| `Scenario_09/00126` | 34.28 | 0.062696 | no |
| `Scenario_09/00127` | 60.03 | 0.048090 | no |
| `Scenario_09/00128` | 26.38 | 0.081010 | no |
| `Scenario_09/00129` | 17.08 | 0.159668 | no |
| `Scenario_09/00130` | 17.06 | 0.133615 | no |
| `Scenario_09/00131` | 15.18 | 0.062818 | no |
| `Scenario_09/00132` | 19.78 | 0.144406 | no |
| `Scenario_09/00133` | 26.89 | 0.109549 | no |
| `Scenario_09/00134` | 21.63 | 0.127438 | no |
| `Scenario_09/00135` | 11.84 | 0.092590 | no |
| `Scenario_09/00136` | 46.27 | 0.106332 | no |
| `Scenario_09/00137` | 26.10 | 0.087315 | no |
| `Scenario_09/00138` | 32.97 | 0.064357 | no |
| `Scenario_09/00139` | 12.86 | 0.182082 | no |
| `Scenario_09/00140` | 65.66 | 0.087197 | no |
| `Scenario_09/00141` | 18.47 | 0.145988 | no |
| `Scenario_09/00142` | 22.02 | 0.144689 | no |
| `Scenario_09/00143` | 95.57 | 0.083913 | no |
| `Scenario_09/00144` | 55.01 | 0.170769 | no |
| `Scenario_09/00145` | 43.01 | 0.179993 | no |
| `Scenario_09/00146` | 17.24 | 0.164386 | no |
| `Scenario_09/00147` | 15.38 | 0.173592 | no |
| `Scenario_09/00148` | 81.53 | 0.142479 | no |
| `Scenario_09/00149` | 9.73 | 0.163496 | sí |
| `Scenario_09/00150` | 14.36 | 0.168492 | no |
| `Scenario_09/00151` | 67.01 | 0.291107 | no |
| `Scenario_09/00152` | 10.69 | 0.142527 | no |
| `Scenario_09/00153` | 20.51 | 0.084032 | no |
| `Scenario_09/00154` | 41.54 | 0.093880 | no |
| `Scenario_09/00155` | 11.55 | 0.179299 | no |
| `Scenario_09/00156` | 19.81 | 0.097149 | no |
| `Scenario_09/00157` | 69.25 | 0.201924 | no |
| `Scenario_09/00158` | 2.83 | 0.078435 | sí |
| `Scenario_09/00159` | 43.05 | 0.089061 | no |
| `Scenario_09/00160` | 29.32 | 0.067120 | no |
| `Scenario_09/00161` | 32.78 | 0.460321 | no |
| `Scenario_09/00162` | 52.06 | 0.053644 | no |
| `Scenario_09/00163` | 30.94 | 0.247220 | no |
| `Scenario_09/00164` | 14.99 | 0.061459 | no |
| `Scenario_09/00165` | 55.29 | 0.334150 | no |
| `Scenario_09/00166` | 93.33 | 0.036858 | no |
| `Scenario_09/00167` | 86.35 | 0.042600 | no |
| `Scenario_09/00168` | 5.56 | 0.113743 | sí |
| `Scenario_09/00169` | 83.98 | 0.136765 | no |
| `Scenario_09/00170` | 88.21 | 0.123544 | no |
| `Scenario_09/00171` | 115.41 | 0.071757 | no |
| `Scenario_09/00172` | 38.56 | 0.302345 | no |
| `Scenario_09/00173` | 19.06 | 0.048207 | no |
| `Scenario_09/00174` | 10.57 | 0.186021 | no |
| `Scenario_09/00175` | 3.57 | 0.154233 | sí |
| `Scenario_09/00176` | 27.88 | 0.108336 | no |
| `Scenario_09/00177` | 36.48 | 0.179533 | no |
| `Scenario_09/00178` | 10.02 | 0.254960 | no |
| `Scenario_09/00179` | 23.72 | 0.099506 | no |
| `Scenario_09/00180` | 149.95 | 0.065260 | no |
| `Scenario_09/00181` | 38.24 | 0.125047 | no |
| `Scenario_09/00182` | 77.09 | 0.100334 | no |
| `Scenario_10/00000` | 63.66 | 0.110226 | no |
| `Scenario_10/00001` | 15.83 | 0.075948 | no |
| `Scenario_10/00002` | 44.97 | 0.056657 | no |
| `Scenario_10/00003` | 36.88 | 0.012355 | no |
| `Scenario_10/00004` | 95.31 | 0.024475 | no |
| `Scenario_10/00005` | 124.11 | 0.010274 | no |
| `Scenario_10/00006` | 47.84 | 0.007032 | no |
| `Scenario_10/00007` | 88.81 | 0.004709 | no |
| `Scenario_10/00008` | 59.41 | 0.046545 | no |
| `Scenario_10/00009` | 46.08 | 0.023527 | no |
| `Scenario_10/00010` | 83.61 | 0.006749 | no |
| `Scenario_10/00011` | 155.45 | 0.004962 | no |
| `Scenario_10/00012` | 269.54 | 0.003524 | no |
| `Scenario_10/00013` | 461.39 | 0.005445 | no |
| `Scenario_10/00014` | 35.84 | 0.017706 | no |
| `Scenario_10/00015` | 113.70 | 0.006880 | no |
| `Scenario_10/00016` | 82.81 | 0.006928 | no |
| `Scenario_10/00017` | 480.69 | 0.002693 | no |
| `Scenario_10/00018` | 97.70 | 0.004966 | no |
| `Scenario_10/00019` | 119.61 | 0.010364 | no |
| `Scenario_10/00020` | 629.28 | 0.002484 | no |
| `Scenario_10/00021` | 602.99 | 0.004997 | no |
| `Scenario_10/00022` | 579.24 | 0.003332 | no |
| `Scenario_10/00023` | 122.33 | 0.007605 | no |
| `Scenario_10/00024` | 73.91 | 0.004996 | no |
| `Scenario_10/00025` | 61.89 | 0.005424 | no |
| `Scenario_10/00026` | 171.11 | 0.003165 | no |
| `Scenario_10/00027` | 251.60 | 0.003652 | no |
| `Scenario_10/00028` | 269.92 | 0.006811 | no |
| `Scenario_10/00029` | 177.06 | 0.017617 | no |
| `Scenario_10/00030` | 36.86 | 0.010855 | no |
| `Scenario_10/00031` | 35.40 | 0.026858 | no |
| `Scenario_10/00032` | 146.12 | 0.022293 | no |
| `Scenario_10/00033` | 51.59 | 0.009398 | no |
| `Scenario_10/00034` | 28.79 | 0.031895 | no |
| `Scenario_10/00035` | 34.99 | 0.011068 | no |
| `Scenario_10/00036` | 48.39 | 0.058002 | no |
| `Scenario_10/00037` | 68.87 | 0.039008 | no |
| `Scenario_10/00038` | 10.00 | 0.052460 | sí |
| `Scenario_10/00039` | 56.06 | 0.018810 | no |
| `Scenario_10/00040` | 66.48 | 0.012002 | no |
| `Scenario_10/00041` | 101.09 | 0.025146 | no |
| `Scenario_10/00042` | 64.78 | 0.021779 | no |
| `Scenario_10/00043` | 9.96 | 0.032291 | sí |
| `Scenario_10/00044` | 82.71 | 0.034002 | no |
| `Scenario_10/00045` | 113.96 | 0.005407 | no |
| `Scenario_10/00046` | 55.30 | 0.027078 | no |
| `Scenario_10/00047` | 33.85 | 0.022611 | no |
| `Scenario_10/00048` | 46.83 | 0.025670 | no |
| `Scenario_10/00049` | 88.08 | 0.123181 | no |
| `Scenario_10/00050` | 52.03 | 0.076462 | no |
| `Scenario_10/00051` | 92.07 | 0.032786 | no |
| `Scenario_10/00052` | 72.79 | 0.058023 | no |
| `Scenario_10/00053` | 48.56 | 0.029226 | no |
| `Scenario_10/00054` | 34.98 | 0.068619 | no |
| `Scenario_10/00055` | 20.51 | 0.105768 | no |
| `Scenario_10/00056` | 62.65 | 0.029466 | no |
| `Scenario_10/00057` | 25.72 | 0.055415 | no |
| `Scenario_10/00058` | 34.96 | 0.188866 | no |
| `Scenario_10/00059` | 45.41 | 0.089237 | no |
| `Scenario_10/00060` | 7.05 | 0.035778 | sí |
| `Scenario_10/00061` | 25.87 | 0.053981 | no |
| `Scenario_10/00062` | 88.87 | 0.111124 | no |
| `Scenario_10/00063` | 45.06 | 0.091209 | no |
| `Scenario_10/00064` | 57.98 | 0.094766 | no |

## Gate propuesto para T2-R10

- **M1:** recall top-1% ≥ **50.0%** (y percentil de rank mediano menor que **81.836%**).
- **M2:** recall ≤10 px ≥ **50.0%** y distancia mediana menor que **37.12 px**.
- **M3:** correlación confianza/corrección ≥ **0.20**. Si M2 no contiene ambas clases,
  M3 queda no identificable y debe repetirse sobre un fold con varios escenarios.

Los márgenes de recall exigen al menos +10 puntos porcentuales sobre el baseline local, con el piso
pre-registrado de 50% de la literatura. Dado `n=5`, estos umbrales son provisionales y deben aprobarse
o recalcularse al disponer del fold real.


## Control OCT-shuffle

Mismo fold y GT; únicamente el OCT se rota un caso, sin auto-pares.

| Métrica | Paired | OCT-shuffle | Delta |
|---|---:|---:|---:|
| M1 top-1% | 0.003402 | 0.002457 | 0.000945 |
| M2 <=10 px | 0.068548 | 0.004032 | 0.064516 |
| M2 mediana px | 37.121 | 107.296 | -70.175 |
