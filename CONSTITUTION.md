# Constitución — FIDO Challenge

Reglas inviolables del proyecto. Si algo en un plan, un prompt o una prisa
contradice esto, gana esto.

---

## I. Integridad de la evaluación

1. **El código de evaluación es intocable.** `vendor/fido/` es una copia
   congelada del repo oficial (commit `5a84d17`). Nadie lo edita. Si hace falta
   otro comportamiento, se envuelve desde `eval/`, no se modifica el original.
   La única mutación permitida es redirigir `CHALLENGE_DATA_DIR` en tiempo de
   ejecución, que es lo que hace el propio `run_local.py` de los organizadores.

2. **Nada sube a Codabench sin pasar `eval/run_local.py`.** Sin excepciones,
   sin "es un cambio chiquito". Hay 20 submissions en toda la fase Competition.

3. **El Mock Test es conjunto de validación, no de entrenamiento.** No se
   entrena, ni se afina, ni se seleccionan hiperparámetros mirando sus
   etiquetas más allá del score agregado. Es el único proxy honesto del test
   oculto que existe.

4. **Ningún resultado se reporta sin la corrida que lo produjo.** Un número sin
   su comando, su commit y su log no es un resultado, es una anécdota.

## II. Honestidad experimental

5. **Se registran los rungs fallidos igual que los exitosos.** `ATTACK_LADDER.md`
   documenta qué se probó y qué pasó, incluyendo lo que no funcionó. Un rung sin
   resultado registrado se considera no ejecutado.

6. **Una hipótesis por rung.** Si un cambio mueve el score y trae tres cosas
   nuevas, no se aprendió nada. Cambios acumulables, medidos uno por uno.

7. **Se pre-registra la expectativa.** Antes de correr un rung se escribe qué
   score se espera y por qué. Sirve para distinguir "funcionó" de "salió
   cualquier cosa y la racionalicé después".

## III. Reproducibilidad

8. **Semillas fijas y declaradas.** Toda corrida de entrenamiento fija su
   semilla y la registra junto con el resultado.

9. **El entorno se declara, no se recuerda.** `infra/bootstrap_pod.sh` es la
   única fuente de verdad del entorno. Si algo se instaló a mano en un pod y no
   está en el script, no existe.

10. **La 5090 es Blackwell (`sm_120`) y exige PyTorch ≥ 2.7 con CUDA 12.8.**
    Los wheels `cu121`/`cu124` importan sin quejarse y truenan en la primera
    operación CUDA. El bootstrap verifica que `sm_120` esté en
    `torch.cuda.get_arch_list()` y lanza un kernel real como prueba.

## IV. Secretos y datos

11. **`.env` nunca se commitea.** Está en `.gitignore`. Las credenciales que
    aparecen ahí son de RunPod y Codabench, y la API key de RunPod puede gastar
    dinero. Si se filtran, se rotan.

12. **El dataset es CC BY-NC-ND.** Uso no comercial, sin redistribución
    modificada. No se sube a repos públicos, ni a Hugging Face, ni a un bucket
    abierto.

13. **Los datos crudos no entran a git.** `data/`, pesos y zips están
    gitignorados. Viven en el volumen de RunPod y en disco local.

## V. Alcance y prioridades

14. **Task 2 primero.** El campo está más débil ahí (al 2026-08-14: líder en
    0.475, segundo en 0.296, cuatro equipos en 0.000). Task 1 está apretado
    (0.619 contra 0.618) y rinde menos por hora invertida.

15. **La fecha que importa es el 4 de septiembre de 2026**, cierre de Final
    Round. La fase Competition (cierre 20 de agosto) sirve para calibrar contra
    el leaderboard, no para ganar.

16. **En la Final Round el límite real es el timeout global de 600 s** para 100
    casos, o sea 6 s por caso de promedio — tres veces más estricto que el
    límite nominal de 20 s por caso. Cualquier diseño que dependa de gastar los
    20 s completos está muerto en la ronda que cuenta.

## VI. Colaboración

17. **Hasta 2 coautores por equipo** en el paper conjunto de los organizadores.
    Definir quiénes antes de la entrega final, no después.

18. **Los organizadores coordinan un paper conjunto** dentro de los 6 meses
    siguientes. Publicación individual permitida después, citando el manuscrito
    oficial. Cualquier uso posterior de estos resultados respeta eso.
