# Task 2 canonical-v1 submission template

Este directorio no es una submission ni contiene un solver/checkpoint. El
pipeline entrenado debe implementar `predict_canonical(oct_volume, opmi_image)`
y envolverse con `fido.inference_task2.CanonicalTask2Bundle`.

El entry point obligatorio es `fido.inference_task2.inference`: convierte
`M_canonical` a la matriz oficial nativa mediante `M_native=M_canonical@C`
inmediatamente antes de retornar `float64 (3,3)`. Al empaquetar una submission
real, este módulo pequeño debe copiarse dentro del zip autocontenido; no se
modifican las submissions legacy r03/r04.
