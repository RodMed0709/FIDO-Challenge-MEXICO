# `ledger/` — la fuente de verdad del proyecto

Un archivo YAML por hecho. Nada de bases de datos, nada de dashboards: si el
código se rompe, `cat ledger/runs/*.yaml` sigue contando toda la historia.

**Los `.md` de estado se generan desde aquí.** `python -m fido.ledger render`
escribe `ledger/_rendered/{NOW,RESULTS,THE_MAP}.md`. Nadie los edita a mano:
el siguiente render se lleva los cambios.

---

## Los cinco tipos

| carpeta | qué guarda | la pregunta que responde |
|---|---|---|
| `runs/` | una corrida de entrenamiento con números | ¿de dónde salió este 0.8534? |
| `defects/` | defectos conocidos, abiertos o cerrados | ¿qué me impide creerme ese número? |
| `facts/` | hechos medidos del entorno o los datos | ¿esto sigue siendo cierto hoy? |
| `decisions/` | decisiones tomadas | ¿por qué hacemos esto, y qué lo revertiría? |
| `assets/` | checkpoints y caches | ¿dónde está el peso, y sigue vivo? |

El esquema completo de cada tipo, campo por campo, está comentado en
`src/fido/ledger.py` (diccionario `ESQUEMA`). Los campos que no son obvios:

- **`revisar_si`** (facts y decisions) — la condición, en prosa, que invalida el
  registro. Se escribe *antes*, no después. `check` la saca a la superficie
  cuando el registro cumple su `vida_util_dias`.
- **`defectos_conocidos`** (runs) — los defectos que hay que tener en la cabeza
  al leer esas métricas. Es lo que impide que `val_distance_auc=0.5681` se cite
  sin su asterisco.
- **`activo`** (assets) — este es el peso que se usa AHORA para esa
  task+componente. Solo puede haber uno; `check` avisa si hay dos.
- **`existe`** / **`verificado_el`** (assets) — liveness. Los checkpoints viven
  en un pod efímero: `desconocido` es el valor honesto hasta que alguien lo
  comprueba por SSH.

---

## Los cuatro comandos

```bash
python -m fido.ledger status          # qué corre, qué duele, qué peso usar
python -m fido.ledger check           # podredumbre y contradicciones (exit 1 si hay errores)
python -m fido.ledger best --task 2   # qué checkpoint uso para Task N, y con qué peros
python -m fido.ledger render          # regenera ledger/_rendered/*.md
```

`check` devuelve **exit 1** si encuentra errores, así que sirve como puerta:
correrlo antes de preparar una submission cuesta un segundo.

---

## Registrar una corrida

No se escribe a mano. Los `train_*.py` ya llaman a `record_run_safe()` al
terminar, y la procedencia se captura sola: commit de git y si el árbol está
sucio, `sys.argv` completo, semilla, hostname, plataforma, timestamps, duración,
y el tamaño y SHA-256 del checkpoint.

```python
from fido import ledger

inicio = ledger.ahora()          # al arrancar
...
ledger.record_run_safe(          # al terminar; nunca levanta una excepción
    id=ledger.id_de_corrida("p40-t2-registracion", inicio),
    titulo="T2 registración: heatmap por correlación",
    peldano="40-t2-decodificador-corregido",
    script="fido.train.train_task2_baseline",
    task=2,
    metricas={"val_corner_auc": 0.0048, "val_mean_error_px": 66.25},
    epoca=60, epocas_planeadas=60, n_val=248, fold=0, n_folds=5,
    checkpoint="checkpoints/task2_fixed/model_1.pth",
    semilla=0, inicio=inicio,
)
```

**Limitación conocida**: si la corrida muere (`OSError` de MooseFS, OOM, un
`kill`), nunca llega al final y no se registra. Esas corridas se anotan a mano
—con la misma honestidad que las que terminan— porque un peldaño sin resultado
escrito cuenta como no ejecutado (`CONSTITUTION.md` §5).

---

## Editar a mano

Está permitido y es el plan B deliberado: son archivos de texto. Dos reglas.

1. **No cambies el `id`**: es el nombre del archivo.
2. **Corre `check` después**: valida el esquema de todos los registros y te dice
   exactamente qué campo quedó mal.

Los cambios hechos desde el código quedan anotados en la lista `historial` del
propio registro (campo, valor anterior, valor nuevo, motivo). Un registro no se
borra nunca; se cierra o se marca como revertido.

Para cerrar un defecto sin abrir el YAML:

```python
from fido.ledger import cerrar_defecto
cerrar_defecto("def-t1-distance-auc-inflado", "evaluate() ya penaliza en vez de descartar")
```

Para confirmar que un checkpoint sigue vivo tras verificarlo por SSH, edita su
YAML: `existe: si` y `verificado_el: 2026-08-18`. `check` deja de dar la lata.

---

## Qué NO va aquí

- **Prosa larga y narrativa**: eso vive en el README del peldaño, en
  `experiments/NN-.../`. El ledger guarda hechos y números.
- **Hipótesis y expectativas pre-registradas**: `ATTACK_LADDER.md`.
- **Reglas del proyecto**: `CONSTITUTION.md`. El ledger no manda sobre ellas —
  fíjate en `dec-task2-primero`, que está marcada para revisión pero cuya
  reversión exige cambiar la constitución, no este archivo.
