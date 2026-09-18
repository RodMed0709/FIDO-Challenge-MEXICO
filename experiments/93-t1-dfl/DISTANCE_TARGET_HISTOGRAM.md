# T1-93 -- distribucion real del target oficial de distancia (px)

**Fuente:** `data/_annotations/Task 1/Scenario_01..10` (`gt[2] / 7.8125`, `gt = annotation["Ground Truth"]["Task 1"]`).  
**Filtro:** solo canula activa (`ILM Distance` presente y `< 1e6`), igual que `_cannula_is_active` en `src/fido/data/task1.py`.  
**Frames totales escaneados:** 61,691  
**Frames con canula activa:** 61,691  
**Frames con canula activa pero sin `Ground Truth.Task 1`:** 0  
**n usado para el histograma:** 61,691  
**Tiempo total:** 67.9s  

min=6.704px  max=1178.769px  media=254.429px  std=166.611px  
**Casos con target negativo:** 0 (0.0000%)  

## Percentiles

| percentil | valor (px) |
|---:|---:|
| 0.1 | 24.750 |
| 1 | 42.041 |
| 5 | 60.333 |
| 10 | 76.965 |
| 25 | 136.264 |
| 50 | 216.692 |
| 75 | 337.825 |
| 90 | 465.635 |
| 95 | 576.665 |
| 99 | 812.033 |
| 99.9 | 1131.277 |

## Por escenario (n con target valido)

| escenario | n |
|---|---:|
| Scenario_01 | 6,703 |
| Scenario_02 | 5,537 |
| Scenario_03 | 10,153 |
| Scenario_04 | 4,246 |
| Scenario_05 | 5,160 |
| Scenario_06 | 5,802 |
| Scenario_07 | 6,855 |
| Scenario_08 | 5,599 |
| Scenario_09 | 6,010 |
| Scenario_10 | 5,626 |

## Reproduccion

```bash
python analysis/measure_task1_distance_target_histogram.py
```
