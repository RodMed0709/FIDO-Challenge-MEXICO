# T1-93 -- fraccion real de NaN de `distance_from_segmentation`

**Checkpoint:** `<repo-root>\submissions\r06-fallback-fixed\model_0.pth`  
**Muestra:** sistematica, objetivo 60 casos/escenario, canula activa, con al menos un B-scan presente en el zip.  
**Total muestreado:** 600  
**Medido (finito):** 449  
**NaN (fallback):** 151  
**Fraccion NaN:** 0.2517 (25.2%)  
**Tiempo total:** 2091.8s  

## Desglose por motivo (sobre los NaN)

| motivo | n | fraccion del total muestreado |
|---|---:|---:|
| sin_instrumento | 150 | 0.2500 |
| sin_ilm_en_ventana | 1 | 0.0017 |

## Por escenario

| escenario | n activos (total) | n muestreado | n medido | n NaN | fraccion NaN |
|---|---:|---:|---:|---:|---:|
| Scenario_01 | 6,703 | 60 | 39 | 21 | 0.3500 |
| Scenario_02 | 5,537 | 60 | 54 | 6 | 0.1000 |
| Scenario_03 | 10,153 | 60 | 48 | 12 | 0.2000 |
| Scenario_04 | 4,246 | 60 | 55 | 5 | 0.0833 |
| Scenario_05 | 5,160 | 60 | 51 | 9 | 0.1500 |
| Scenario_06 | 5,802 | 60 | 57 | 3 | 0.0500 |
| Scenario_07 | 6,855 | 60 | 44 | 16 | 0.2667 |
| Scenario_08 | 5,599 | 60 | 1 | 59 | 0.9833 |
| Scenario_09 | 6,010 | 60 | 46 | 14 | 0.2333 |
| Scenario_10 | 5,626 | 60 | 54 | 6 | 0.1000 |

## Reproduccion

```bash
python analysis/measure_task1_segmentation_nan_fraction.py
```
