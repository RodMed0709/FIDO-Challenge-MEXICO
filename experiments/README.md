# experiments/ — convención

Un directorio por peldaño, numerado en orden de ejecución. Mismo patrón que se
usó en ORENA_CHALLENGE, que funcionó bien.

```
experiments/
├── 00-smoke-contract/          el peldaño 00 siempre es "¿el contrato funciona?"
│   ├── README.md               la escalera, el veredicto, por qué existe
│   ├── RESULTS.csv             una fila por corrida, columnas anchas
│   ├── runs/                   logs y checkpoints por corrida
│   └── _tools/                 scripts propios de este peldaño
├── 10-t1-distancia-unet/
├── 20-t2-puente-vasos/
└── ...
```

## Reglas

**Numeración por decenas.** Los peldaños van `00`, `10`, `20`, `30`, `40`, `50`…
El hueco es deliberado: cuando hace falta un peldaño intermedio (un control, una
verificación que desbloquea al siguiente) entra como `05` o `15` sin renumerar
nada ni romper las referencias cruzadas ya escritas. Los sufijos de corrida
siguen al peldaño: `40a`, `40b`.

**Un peldaño, una hipótesis.** Si un cambio mueve el score y trae tres cosas
nuevas, no se aprendió nada. Cambios acumulables, medidos uno por uno.

**Pre-registrar la expectativa.** Antes de correr, el README dice qué score se
espera y por qué. Sirve para distinguir "funcionó" de "salió cualquier cosa y lo
racionalicé después".

**Todo peldaño declara su control.** Contra qué corrida previa se compara. Sin
control, un número no significa nada.

**Los fracasos se escriben igual.** Un 🔴 NO-GO con su costo es información
valiosa: evita que alguien (o yo mismo dentro de tres días) vuelva a intentarlo.

**El costo va en el README.** Cuántos dólares de GPU costó el peldaño.

## Formato del README de un peldaño

```markdown
# NN — título: la pregunta que responde

## Escalera
| run | qué es | métrica |
|---|---|---|
| ../MM-.../ run X | el control | 0.xxxx |
| NNa | lo nuevo | **0.xxxx** |

## Expectativa pre-registrada
Qué esperábamos y por qué. Escrito ANTES de correr.

## 🔴/🟢 RESULTADO (fecha)
El veredicto en una frase, y luego el detalle. Costo incluido.

## Por qué existe este peldaño
Qué peldaño anterior lo desbloqueó o lo hizo necesario.
```

## RESULTS.csv

Columnas mínimas: `run, config, score_local, score_codabench, fecha, notas`.
Las notas son largas a propósito: ahí va todo lo que se mantuvo fijo respecto al
control, que es lo que hace interpretable la comparación.

## Relación con `ledger/`

El README de un peldaño cuenta la **historia**: qué se preguntaba, qué se
esperaba, qué salió y qué se aprendió. Los **números** no viven aquí: viven en
`ledger/runs/*.yaml`, con su comando, su commit y su procedencia capturada
automáticamente (`CONSTITUTION.md` §4). Cada corrida del ledger declara a qué
peldaño pertenece en su campo `peldano`, con el nombre exacto de esta carpeta.

Para ver los números de un peldaño sin abrir los YAML:

```bash
python -m fido.ledger status            # qué corre, qué duele, qué peso usar
python -m fido.ledger render            # regenera las vistas en ledger/_rendered/
```

`RESULTS.csv` sigue siendo útil como resumen ancho y legible del peldaño, pero
si un número aparece en los dos sitios, **manda el ledger**.
