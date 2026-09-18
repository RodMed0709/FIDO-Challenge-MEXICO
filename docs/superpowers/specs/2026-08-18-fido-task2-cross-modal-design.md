# Diseño — FIDO Task 2: localización cross-modal y registro 4-DOF

**Fecha:** 2026-08-18  
**Estado:** aprobado por Rodrigo para planificación  
**Sustituye:** `2026-08-14-fido-task2-design.md` cuando exista conflicto

## Objetivo

Localizar qué región, orientación y escala del fundus corresponde al volumen
iOCT y producir una similitud reflejada `3x3`. El sistema debe aprender una
representación común entre modalidades, buscar hipótesis de pose explícitamente
y refinar solo candidatos que ya estén dentro de una cuenca útil.

## Restricciones

- `vendor/fido/` es intocable y toda submission pasa `eval/run_local.py`.
- GroupKFold por escenario; Mock Test solo como score agregado final.
- Los rangos exactos observados previamente en Mock quedan en cuarentena: no
  seleccionan bins, augmentación, arquitectura ni hiperparámetros.
- Cada peldaño cambia una sola hipótesis.
- La métrica de modelo es `corner_auc` oficial 0..10, pooled y por escenario.
  Retrieval, NCC, MI y errores por componente son gates diagnósticos.
- El artefacto final devuelve `float64 (3,3)`, maneja `oct_volume=None`, usa
  `model_1.pth` y completa 100 casos en `<=540 s`.

## Estado actual

- El baseline convierte el volumen en `mean(axis=1)`, usa dos CNN y desliza la
  plantilla OCT mediante correlación sobre el fundus.
- Su representación no corresponde: top-1% del positivo `0.34-0.39%`, centro
  `<=10 px` en `6.9%`, mediana `37.12 px`; la confianza no predice aciertos.
- Intensidades crudas, NCC/MI, vasos e instrumento como solución principal
  quedaron refutados sobre train. No se repiten.
- DINOv2 se probó congelado, con backbone compartido entre RGB y OCT. Ese
  experimento no refuta encoders independientes fine-tuned.
- La escala está fuertemente asociada al escenario. Debe medirse causalmente
  con sustitución GT y augmentación como único cambio; un split aleatorio no
  aísla escala de identidad/anatomía.

## Arquitectura objetivo

```text
OCT 3D -> proyector axial -> encoder OCT -----\
                                               > descriptores comunes
fundus RGB -------------> encoder fundus+FPN -/
       -> búsqueda batched de reflexión/rotación/escala/traslación
       -> top-K hipótesis
       -> correspondencias locales
       -> Umeyama/Procrustes reflejado 4-DOF
       -> refinador local opcional
       -> matriz 3x3
```

- Los encoders no comparten pesos.
- La representación se entrena con positivos derivados del GT y hard negatives
  del mismo escenario.
- El fundus se codifica una sola vez; no se genera un OCT por crop.
- La reflexión es una convención fija; no se aprende.
- Rotación y escala se buscan geométricamente para evitar extrapolación por una
  cabeza de regresión.
- DINOv2 se evalúa después de que el gate de representación exista: LP-FT,
  unfreeze y LLRD, comparado contra CNN con el mismo loss y solver.

## Estrategia de validación

1. Descomponer error con sustitución GT de un componente a la vez.
2. Crear augmentación geométrica exacta derivada solo de train.
3. Probar representación común sin solver completo.
4. Solo si el retrieval pasa, añadir búsqueda angular y de escala.
5. Solo si el globalizador tiene AUC no trivial, añadir top-K/refinador.
6. Confirmar en folds adicionales, congelar, perfilar y recién entonces abrir
   el Mock agregado.

## Gates

- Representación: positivo top-1% `>=50%`, centro con pose parcial GT `>=50%`
  a 10 px y colapso claro al barajar OCT.
- Ángulo: mediana `<3 deg`, p90 `<6 deg` con posición/escala GT.
- Escala: error relativo mediano `<=3%`, p90 `<=6%` en un holdout sintético
  generado exclusivamente desde train.
- Globalizador: AUC completo `>0.10` antes de permitir refinamiento.
- Modelo candidato: mejora `>=0.05` AUC o `>=25%` en mediana/p90 sin aumentar
  errores catastróficos; el gate final siempre es AUC.
- Submission: mediana OOF `>=0.20`, al menos tres folds `>0.10`, OCT shuffle
  colapsa, cero timeouts y `<=540 s/100`.

## Fuera de alcance

- GAN/diffusion OCT-fundus, reconstrucción 3D por crop y cientos de forwards.
- DINOv2 frozen/shared, matchers off-the-shelf sin adaptación, e2cnn y
  RetinaRegNet completo.
- DPCN/phase correlation sobre intensidades crudas.
- IHN/refinador como inicializador global.
- Instrumento o crosshair fundus-only como solución principal.
- Elegir rangos usando las etiquetas del Mock Test.

