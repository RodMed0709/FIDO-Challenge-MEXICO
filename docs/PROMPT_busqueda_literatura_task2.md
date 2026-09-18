# Prompt para la próxima sesión — búsqueda de literatura para Task 2

Copia todo lo que está entre las líneas de guiones y pégalo como primer mensaje
de la sesión nueva.

---

Necesito que orquestes una búsqueda de literatura profunda para desbloquear
Task 2 del FIDO Challenge. Usa Codex (`codex exec --skip-git-repo-check`) y
pídele que lance **sus propios sub-agentes en paralelo**, que descargue los PDFs
reales y que los analice — no quiero resúmenes de abstracts.

Antes de escribir el prompt de Codex, lee para tener el contexto exacto:
`ATTACK_LADDER.md` (secciones T2-R2 y T2-R4), `RESULTS.md`, y
`experiments/20-t2-puente-vasos/`, `experiments/30-t2-baseline-roto/`,
`experiments/40-t2-decodificador-corregido/`.

## El problema, en términos precisos

Registrar una **proyección en-face de un volumen OCT** (128×512, escala de
grises) sobre una **foto de fundus de microscopio quirúrgico** (1024×1024 RGB).
La salida es una matriz 3×3 que es una **similitud reflejada de 4 DOF**
(det negativo en el 100% de 1214 casos): traslación, rotación θ~U(0,2π), y
escala ~160±13.7 px. El área común es ~2.4% de la imagen del fundus.

La métrica es AUC del error de esquinas sobre umbrales enteros 0..10 px. O sea:
**un caso solo puntúa si su error de esquina baja de 10 px.** Líder actual del
leaderboard: 0.475. Nosotros: 0.0048.

## Lo que YA está medido y refutado — no lo propongas otra vez

Esto no son opiniones, son mediciones con evidencia guardada en el repo:

1. **Template matching clásico sobre máscaras de vasos: REFUTADO por oráculo.**
   Tomando los parámetros REALES del ground truth (sin ninguna búsqueda), la
   correlación NCC entre el patch de vasos del en-face alineado en la posición
   correcta y la región real del fundus da **≈0 (ruido puro)** en 4 de 4 casos.
   Ningún buscador puede mejorar eso porque no hay nada que encontrar. Se
   probaron además binarizado, MIP y dilatación de 2 a 21 px: ninguna versión
   pasa de NCC 0.018.
2. **DINOv2 (ViT-S/14) congelado como backbone compartido: FALLÓ.** 40 épocas,
   terminó en 136.6 px de error medio contra 66.25 px del encoder CNN casero, y
   **empeoró** con el entrenamiento (100 → 110 → 125 → 136). Interpretación:
   sus features no codifican la correspondencia entre una foto de retina y una
   proyección de OCT, lo cual es coherente con que se preentrenó en fotos
   naturales.
3. **Traducción OCT→fundus con GAN antes de matchear:** ya descartado por la
   literatura previa del proyecto.

## Dónde está el error, medido por componente

Sustituyendo cada componente por su valor real del GT, una a una:

| componente | cuánto del error explica | error propio |
|---|---|---|
| **posición (tx,ty)** | **65.6%** | mediana 32.9 px |
| rotación (θ) | 12.5% | mediana 5.7° |
| escala (s) | 5.4% | mediana 7.0% |

Y el dato que más importa: **el pico del mapa de correlación cae cerca del
centro real en solo el 28.6% de los casos**, aunque el mapa está bien
pronunciado (el pico se lleva 9.1% de la masa softmax contra 0.024% uniforme).

Traducción: el modelo está **seguro y equivocado**. No es un problema de
decodificación (probé argmax duro, soft-argmax local y temperatura baja: todos
dan 54-63 px, o sea ~14% de mejora, no el 10x que hace falta) ni de rotación.
**Es que las representaciones de las dos modalidades no se corresponden.**

Techo estructural conocido: con posición, rotación y escala perfectas, la
parametrización actual da AUC 0.639 (el resto es redondeo del round-trip).

## Qué quiero que investigue

La pregunta central: **¿qué arquitecturas resuelven correspondencia cross-modal
cuando las dos modalidades no comparten apariencia?** En concreto:

- Registración multimodal de retina (fundus ↔ OCT / OCTA / SLO / angiografía) —
  es un área de investigación real y activa, con datasets y benchmarks propios.
- Matching sin detector (LoFTR, ASpanFormer, MatchFormer, RoMa, DKM y sucesores):
  qué tan bien transfieren a pares cross-modal y qué modificaciones necesitan.
- Aprendizaje de representaciones contrastivas para registración cross-modal
  (CoMIR y lo que vino después): entrenar explícitamente un espacio común en vez
  de esperar que dos encoders converjan solos.
- Correlación de fase diferenciable y métodos log-polar (DPCN++, IHN y
  sucesores), que desacoplan rotación/escala/traslación por construcción.
- Backbones equivariantes a rotación, dado que θ es uniforme en [0,2π) y solo
  tenemos 10 ojos de entrenamiento.
- Cualquier cosa que ataque directamente el modo de fallo medido: el mapa de
  similitud pica con confianza en el lugar equivocado.

## Restricciones duras que debe respetar toda recomendación

- **Presupuesto de inferencia**: 600 s para 100 casos en la ronda final = 6 s por
  caso, cargando un volumen de 128 PNGs incluido. Un método que necesite 30 s por
  caso está muerto aunque sea perfecto.
- **Datos**: 1214 pares de entrenamiento, de solo **10 ojos**. Splits por
  GroupKFold por escenario. Cualquier método hambriento de datos tiene que
  justificar cómo sobrevive a eso.
- **Entorno**: PyTorch ≥2.7 con CUDA 12.8 (RTX 5090, Blackwell sm_120). Librerías
  de investigación poco mantenidas son un riesgo real: `escnn`/`e2cnn` ya se
  marcaron como riesgo por incompatibilidad con PyTorch 2.7.
- **Fecha límite**: la Final Round cierra el 4 de septiembre de 2026. Cualquier
  cosa que no se pueda implementar y entrenar en ~2 semanas no sirve.

## Reglas anti-alucinación — esto es innegociable

Este proyecto ya tuvo agentes de literatura que se ramificaron sin control y
otros que murieron sin escribir nada. Y una cifra clave del proyecto
(arXiv 2603.25555) tuvo que verificarse a mano porque los agentes citan papers
que no existen.

Por lo tanto, exígele a Codex:

1. **Cada paper citado debe tener PDF descargado y leído.** Un paper del que
   solo se leyó el abstract se marca explícitamente `[solo-abstract]` y sus
   cifras se tratan como orden de magnitud, no como benchmark.
2. **Nada de cifras inventadas.** Si un paper reporta un número, debe venir con
   la sección o tabla de donde salió. Si no se encontró el número, se dice "no
   reportado", no se estima.
3. **Escribir salidas temprano y de forma incremental**, no al final. Un agente
   que muere a mitad debe dejar lo que ya encontró.
4. **Tope explícito de búsquedas y de sub-agentes**, y prohibido que los
   sub-agentes lancen sub-agentes propios.
5. Cada recomendación debe decir **contra qué la refutaría**: qué medición la
   descartaría. Sin eso es opinión, no hipótesis.

## Entregable

Un archivo `literature/04_task2_cross_modal.md` con:

- Las 3-5 arquitecturas candidatas, ordenadas por *retorno esperado dividido
  entre riesgo de implementación*, no por qué tan impresionante suena el paper.
- Para cada una: qué problema exacto resuelve del nuestro, qué reporta en su
  paper (con procedencia), qué habría que implementar, cuánto costaría, y qué
  evidencia la refutaría.
- Una sección explícita de **qué NO intentar y por qué**, incorporando lo que ya
  refutamos aquí para que nadie lo reintente en tres semanas.
- Los PDFs descargados en `literature/pdfs/`.

Cuando Codex termine, no aceptes su reporte sin verificar: toma 2 papers al azar
de sus citas y comprueba que existen y que dicen lo que dice que dicen.

Al final, registra los hallazgos como `facts` y `decisions` en el ledger
(`python -m fido.ledger`), y actualiza `ATTACK_LADDER.md` con los peldaños
nuevos que salgan, cada uno con su expectativa pre-registrada.

---
