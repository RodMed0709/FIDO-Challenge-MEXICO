# T2-80 — resultados

## Estado

Implementación y contrato local completos. La corrida numérica fold 0 queda
pendiente en el pod porque este rung no autoriza lanzar trabajos RunPod.

El baseline histórico del mismo checkpoint/fold queda congelado como referencia:
M1 top-1% 0.34% (vecino), M2 <=10 px 6.9%, mediana 37.12 px y M3 -0.017.
No se presentan como resultados nuevos de T2-80.

## Advertencia de provenance

`task2_fixed` usa `SCALE_REF=160`, derivado con conocimiento agregado de los
1214 casos antes del split. El diagnóstico resultante será transductivo y no
puede cerrar un gate OOF limpio. La próxima receta debe ajustar referencias y
rangos solo sobre `train_idx` y persistir los IDs usados.

## Comandos pendientes

Usar exactamente los dos comandos de `PRE_REGISTRATION.md`. Ambos escriben
artefactos por caso; correspondence ejecuta paired y OCT-shuffle sobre el mismo
fold. Ningún resultado autoriza submission a Codabench.
