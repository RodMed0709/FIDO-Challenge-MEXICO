# Diseño — FIDO Task 1: mejora integral

**Fecha:** 2026-08-18  
**Estado:** aprobado por Rodrigo para planificación  
**Sustituye:** el roadmap informal de Task 1 en `ATTACK_LADDER.md` cuando exista conflicto

## Objetivo

Mejorar el artefacto de Task 1 que obtuvo `0.569` en Codabench, preservando la
CNN como control, evaluando DINOv2 con transferencia real y corrigiendo la
componente de distancia. La selección se hará con validación por escenario y
la métrica oficial completa, no con el Mock Test ni con promedios sobre casos
descartados.

## Restricciones

- `vendor/fido/` es intocable.
- Cada cambio científico ocupa un peldaño independiente y se pre-registra en
  `ATTACK_LADDER.md`.
- GroupKFold por escenario; nunca split aleatorio por frame para seleccionar.
- El Mock Test se usa una sola vez sobre un candidato ya congelado y solo para
  score agregado, contrato y tiempo.
- Todo resultado incluye comando, commit, semilla, log y predicciones por caso.
- La inferencia final implementa `inference(task_id, oct_volume, opmi_image,
  model)`, maneja `oct_volume=None` y usa `model_0.pth`.
- El contenedor final debe terminar 100 casos en `<=540 s`, dejando 10% de
  margen frente al timeout de 600 s.

## Estado que debe reproducirse

- CNN desde cero, stride 16, heatmap 64x64 y soft-argmax global: AUC de
  validación histórica `~0.8537`.
- DINOv2 ViT-S/14 congelado y decoder x2: AUC histórica `~0.8005`.
- Esos números no forman un A/B limpio: existe un defecto de fallback que puede
  cruzar el split y la corrida DINO no tiene comando/log completo.
- El `distance_auc=0.6113` está inflado porque la evaluación descartaba casos
  no medibles; no es baseline autoritativo.

## Arquitectura de decisión

Task 1 se divide en dos componentes que no se mezclan hasta el final:

1. **Keypoint (peso 0.7):** sanear medición, comparar decoders sin retraining,
   mejorar resolución de la CNN, y después evaluar DINOv2 mediante LP-FT y
   full fine-tuning con LLRD bajo el mismo head y split.
2. **Distancia (peso 0.3):** medir el techo geométrico y la cobertura real,
   después comparar geometría contra una cabeza distribucional all-case.

El candidato final se elige por `0.7*keypoint_auc + 0.3*distance_auc` calculado
sobre los mismos casos y folds. No se suman mejores números de experimentos
incompatibles.

## Transferencia DINOv2 válida

La conclusión permitida del experimento anterior es únicamente: “DINOv2
congelado con el decoder ensayado perdió”. La prueba completa será:

1. Reproducir el control frozen con provenance.
2. Entrenar el decoder ganador como linear probe.
3. Descongelar todo el backbone.
4. Aplicar LLRD: head `1e-4`, último bloque `1e-5`, decay `0.75` hacia bloques
   tempranos, warmup 5%, cosine, weight decay `0.05` excepto bias/norm.
5. No cambiar decoder, resolución, split ni loss durante el peldaño de unfreeze.

## Gates

- Cada decoder se evalúa sobre los mismos logits/checkpoint cuando sea posible.
- Una mejora de keypoint debe subir AUC 0..10 y PCK@1/PCK@3; bajar solo el error
  medio no basta.
- DINO full FT debe superar LP por `>=0.01` AUC o quedar a `<=0.005` del mejor
  CNN con errores complementarios para seguir vivo.
- La cabeza de distancia debe mejorar el AUC all-case honesto por `>=0.02`.
- Una arquitectura pasa a confirmación solo si mejora el score ponderado
  interno por `>=0.01`.
- La confirmación exige cinco folds y desglose por los diez escenarios.

## Fuera de alcance

- Tuning sobre Mock Test.
- Mezclar local soft-argmax, DARK, UDP y stride 4 en una sola corrida.
- Reabrir DINO con otra resolución dentro del mismo peldaño si hay OOM.
- Ensemble o TTA antes de demostrar complementariedad out-of-fold.
- Modificar la submission histórica `r03-dist-v2`.

