# Evidencia que abre T2-83

Fuente local: `ATTACK_LADDER.md`, T2-R11b. Oráculo sobre los 1214 casos de
entrenamiento, 469 puntos de instrumento y 93,800 nulls. En radios
0.005/0.01/0.02/0.04, `transpose__flip_u__flip_v` obtuvo ratios
3.6364/3.4309/4.1775/3.1568; identity obtuvo
1.1747/1.1561/0.7491/0.6995.

La implementación exacta viene de `transform_uv()` en
`analysis/verify_task2_enface_convention.py`: primero `u=1-u`, `v=1-v` y luego
transpose. Aplicada al array en vez de a puntos equivale a
`native[::-1, ::-1].T`.

La transformación de puntos correspondiente es
`C=[[0,-1,1],[-1,0,1],[0,0,1]]`: `(u_native,v_native)=(1-v,1-u)`.
Por ello orientar solo la imagen sería incorrecto; el GT interno se compone
como `M_native@C`. Los corners proyectados por ambas rutas son idénticos.

El reporte completo original no está materializado localmente. Por eso T2-83
no toma el resumen como cierre: exige repetir el oráculo pre-registrado usando
solo `train_idx` del fold y persistir JSON, IDs hash y tablas completas.
