"""Ledger del proyecto FIDO — la fuente de verdad del estado.

Cinco tipos de registro, uno por archivo YAML, en `ledger/<plural>/<id>.yaml`.
Los `.md` de estado se GENERAN desde aquí (`render`), no se editan a mano.

Por qué existe
--------------
`CONSTITUTION.md` §4 exige que ningún número se reporte sin su comando, su
commit y su log. Escribir eso a mano en tres archivos distintos produce deriva:
el mismo hecho acaba con tres valores y nadie sabe cuál es el vivo. Aquí cada
hecho vive en exactamente un archivo, y la procedencia se captura sola
(`record_run`), que es la única forma de que se capture siempre.

Los cinco tipos
---------------
run       una corrida de entrenamiento con números y su procedencia
defect    un defecto conocido, con estado abierto/cerrado y severidad
fact      un hecho medido del entorno o los datos, con fecha de caducidad
decision  una decisión tomada, con la evidencia que la revertiría
asset     un checkpoint o cache: dónde vive, qué corrida lo produjo, si sigue vivo

Invariantes
-----------
- Un registro nunca se borra. Los cambios se aplican sobre el archivo y quedan
  anotados en su lista `historial` (qué campo, valor anterior, valor nuevo).
- El `id` de un registro es el nombre de su archivo sin extensión. No se cambia.
- Todo sigue siendo legible con `cat` si este módulo se rompe. Eso es el punto.

Uso
---
    python -m fido.ledger status
    python -m fido.ledger check
    python -m fido.ledger best --task 2
    python -m fido.ledger render
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

__all__ = [
    "LedgerError",
    "ledger_dir",
    "ahora",
    "id_de_corrida",
    "escribir_registro",
    "actualizar_registro",
    "cargar",
    "cargar_uno",
    "validar",
    "capturar_procedencia",
    "record_run",
    "record_run_safe",
    "cerrar_defecto",
    "revisar_pendientes",
    "contradicciones",
    "mejor_checkpoint",
    "render_now",
    "render_results",
    "render_the_map",
    "render_todo",
    "main",
]


# --------------------------------------------------------------------------
# Esquemas
# --------------------------------------------------------------------------

TIPOS: dict[str, str] = {
    "run": "runs",
    "defect": "defects",
    "fact": "facts",
    "decision": "decisions",
    "asset": "assets",
}

SEVERIDADES = ("critica", "alta", "media", "baja")
_ORDEN_SEVERIDAD = {s: i for i, s in enumerate(SEVERIDADES)}

# Para cada tipo:
#   orden         → en qué orden se escriben los campos en el YAML (legibilidad)
#   obligatorios  → sin esto el registro no se acepta
#   enums         → valores cerrados
#   fechas        → campos con formato YYYY-MM-DD (None permitido)
ESQUEMA: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------- run --
    # Una corrida de entrenamiento. Los números viven aquí y en ningún otro
    # sitio; RESULTS.md se genera de estos registros.
    #   peldano            carpeta de experiments/ a la que pertenece (decenas)
    #   estado             en_curso | completada | murio | abortada
    #   script             módulo ejecutable, p.ej. fido.train.train_task1_unet
    #   metricas           dict libre {nombre: valor}. Los nombres son los que
    #                      imprime el propio script, sin renombrar.
    #   epoca              época a la que corresponden esas métricas
    #   defectos_conocidos ids de defect que afectan la lectura de estos números
    #   procedencia        bloque capturado por record_run(), no se escribe a mano
    "run": {
        "orden": [
            "id", "tipo", "titulo", "peldano", "estado", "script", "task",
            "metricas", "epoca", "epocas_planeadas", "n_val", "fold", "n_folds",
            "checkpoint", "defectos_conocidos", "procedencia", "notas", "historial",
        ],
        "obligatorios": ["id", "tipo", "titulo", "peldano", "estado", "script", "procedencia"],
        "enums": {
            "estado": ("en_curso", "completada", "murio", "abortada"),
            "task": (1, 2, None),
        },
        "fechas": [],
    },
    # ------------------------------------------------------------- defect --
    # Un defecto conocido del código o del método.
    #   severidad  critica bloquea entrega · alta falsea números · media cuesta
    #              tiempo o dinero · baja incomodidad
    #   afecta     ids de run cuyos números hay que leer a la luz de esto
    "defect": {
        "orden": [
            "id", "tipo", "titulo", "estado", "severidad", "donde", "sintoma",
            "evidencia", "impacto", "afecta", "abierto_el", "cerrado_el",
            "resuelto_por", "notas", "historial",
        ],
        "obligatorios": ["id", "tipo", "titulo", "estado", "severidad", "sintoma", "abierto_el"],
        "enums": {
            "estado": ("abierto", "cerrado"),
            "severidad": SEVERIDADES,
        },
        "fechas": ["abierto_el", "cerrado_el"],
    },
    # --------------------------------------------------------------- fact --
    # Un hecho MEDIDO. Si no se midió, es una decisión o una nota, no un hecho.
    #   ambito          entorno | datos | metrica | competencia
    #   medido_con      comando o script que produjo el valor
    #   revisar_si      condición en prosa que lo invalida
    #   vida_util_dias  a partir de cuántos días `check` lo marca por revisar
    "fact": {
        "orden": [
            "id", "tipo", "titulo", "ambito", "valor", "unidad", "medido_el",
            "medido_con", "revisar_si", "vida_util_dias", "notas", "historial",
        ],
        "obligatorios": ["id", "tipo", "titulo", "ambito", "valor", "medido_el", "revisar_si"],
        "enums": {"ambito": ("entorno", "datos", "metrica", "competencia")},
        "fechas": ["medido_el"],
    },
    # ----------------------------------------------------------- decision --
    # Una decisión tomada, con lo que la revertiría escrito por adelantado.
    #   estado      vigente | revisar | revertida
    #   revisar_si  qué evidencia concreta obliga a reabrirla
    #   evidencia   ids de fact o run en los que se apoya
    "decision": {
        "orden": [
            "id", "tipo", "titulo", "estado", "decision", "porque", "revisar_si",
            "evidencia", "decidido_el", "revisado_el", "vida_util_dias", "notas",
            "historial",
        ],
        "obligatorios": ["id", "tipo", "titulo", "estado", "decision", "porque",
                         "revisar_si", "decidido_el"],
        "enums": {"estado": ("vigente", "revisar", "revertida")},
        "fechas": ["decidido_el", "revisado_el"],
    },
    # -------------------------------------------------------------- asset --
    # Un checkpoint o un cache. Existe para que ningún peso quede huérfano.
    #   clase        checkpoint | cache
    #   componente   qué parte del score produce: keypoint | distancia |
    #                registracion | segmentacion | datos
    #   activo       es el que se usa AHORA para esa task+componente
    #   efimero      vive en un pod que puede desaparecer
    #   existe       si | no | desconocido  (liveness; se verifica fuera de aquí)
    "asset": {
        "orden": [
            "id", "tipo", "titulo", "clase", "ruta", "host", "task", "componente",
            "activo", "efimero", "existe", "verificado_el", "producido_por",
            "bytes", "sha256", "notas", "historial",
        ],
        "obligatorios": ["id", "tipo", "titulo", "clase", "ruta", "host", "efimero", "existe"],
        "enums": {
            "clase": ("checkpoint", "cache"),
            "existe": ("si", "no", "desconocido"),
            "task": (1, 2, None),
            "componente": ("keypoint", "distancia", "registracion", "segmentacion",
                           "datos", None),
        },
        "fechas": ["verificado_el"],
    },
}

_RE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_RE_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")

VIDA_UTIL_POR_DEFECTO = 30  # días, si el registro no declara la suya


class LedgerError(RuntimeError):
    """Un registro no cumple el esquema, o se intentó pisar uno existente."""


# --------------------------------------------------------------------------
# Localización y E/S
# --------------------------------------------------------------------------

def ledger_dir(explicito: str | Path | None = None) -> Path:
    """Devuelve la raíz del ledger.

    Orden de resolución: argumento explícito → `$FIDO_LEDGER_DIR` → la carpeta
    `ledger/` del repo (buscando hacia arriba desde este archivo y desde el cwd)
    → `./ledger`.
    """
    if explicito is not None:
        return Path(explicito)
    env = os.environ.get("FIDO_LEDGER_DIR")
    if env:
        return Path(env)
    for origen in (Path(__file__).resolve(), Path.cwd().resolve() / "_"):
        for padre in origen.parents:
            candidato = padre / "ledger"
            if candidato.is_dir():
                return candidato
    return Path.cwd() / "ledger"


def _dir_tipo(tipo: str, raiz: Path) -> Path:
    if tipo not in TIPOS:
        raise LedgerError(f"tipo desconocido: {tipo!r} (esperaba uno de {sorted(TIPOS)})")
    return raiz / TIPOS[tipo]


def _ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hoy() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ahora() -> str:
    """Timestamp UTC en el formato que usa el ledger. Guárdalo al arrancar."""
    return _ahora()


def id_de_corrida(peldano: str, inicio: str | None = None) -> str:
    """Construye un id de corrida único y legible: `run-<fecha>-<hhmm>-<peldano>`.

    Lleva la hora porque dos corridas del mismo peldaño el mismo día es lo
    normal, y un id repetido pisaría el registro anterior.
    """
    marca = inicio or _ahora()
    fecha, hhmm = marca[:10], marca[11:13] + marca[14:16]
    limpio = re.sub(r"[^a-z0-9-]+", "-", peldano.lower()).strip("-")
    return f"run-{fecha}-{hhmm}-{limpio}"


def _dias_desde(fecha: str | None) -> int | None:
    """Días transcurridos desde una fecha `YYYY-MM-DD` (o `None` si no aplica)."""
    if not fecha:
        return None
    texto = str(fecha)[:10]
    if not _RE_FECHA.match(texto):
        return None
    try:
        d = datetime.strptime(texto, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - d).days


def validar(registro: dict[str, Any]) -> list[str]:
    """Devuelve la lista de problemas del registro. Vacía = válido."""
    problemas: list[str] = []
    tipo = registro.get("tipo")
    if tipo not in ESQUEMA:
        return [f"tipo inválido: {tipo!r}"]
    esquema = ESQUEMA[tipo]

    rid = registro.get("id")
    if not isinstance(rid, str) or not _RE_ID.match(rid):
        problemas.append(f"id inválido: {rid!r} (minúsculas, dígitos, '.', '-', '_')")

    for campo in esquema["obligatorios"]:
        valor = registro.get(campo)
        if valor is None or (isinstance(valor, (str, list, dict)) and len(valor) == 0):
            problemas.append(f"falta el campo obligatorio '{campo}'")

    for campo, permitidos in esquema["enums"].items():
        if campo in registro and registro[campo] not in permitidos:
            problemas.append(
                f"'{campo}' = {registro[campo]!r} no está en {list(permitidos)}"
            )

    for campo in esquema["fechas"]:
        valor = registro.get(campo)
        if valor is not None and not _RE_FECHA.match(str(valor)):
            problemas.append(f"'{campo}' = {valor!r} no tiene formato YYYY-MM-DD")

    desconocidos = set(registro) - set(esquema["orden"])
    if desconocidos:
        problemas.append(f"campos fuera del esquema: {sorted(desconocidos)}")

    return problemas


def _ordenar(registro: dict[str, Any]) -> dict[str, Any]:
    orden = ESQUEMA[registro["tipo"]]["orden"]
    salida = {k: registro[k] for k in orden if k in registro}
    salida.update({k: v for k, v in registro.items() if k not in salida})
    return salida


def _volcar(registro: dict[str, Any], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    texto = yaml.safe_dump(
        _ordenar(registro),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=88,
    )
    destino.write_text(texto, encoding="utf-8")


def escribir_registro(
    tipo: str,
    campos: dict[str, Any],
    *,
    raiz: str | Path | None = None,
    sobrescribir: bool = False,
) -> Path:
    """Crea un registro nuevo. Falla si ya existe (salvo `sobrescribir=True`)."""
    raiz_p = ledger_dir(raiz)
    registro = dict(campos)
    registro["tipo"] = tipo
    registro.setdefault("historial", [])
    problemas = validar(registro)
    if problemas:
        raise LedgerError(
            f"registro {tipo}/{campos.get('id')!r} inválido:\n  - " + "\n  - ".join(problemas)
        )
    destino = _dir_tipo(tipo, raiz_p) / f"{registro['id']}.yaml"
    if destino.exists() and not sobrescribir:
        raise LedgerError(f"ya existe {destino}; usa actualizar_registro() para cambiarlo")
    _volcar(registro, destino)
    return destino


def cargar(tipo: str, *, raiz: str | Path | None = None) -> list[dict[str, Any]]:
    """Lee todos los registros de un tipo, ordenados por id."""
    carpeta = _dir_tipo(tipo, ledger_dir(raiz))
    if not carpeta.is_dir():
        return []
    registros: list[dict[str, Any]] = []
    for archivo in sorted(carpeta.glob("*.yaml")):
        datos = yaml.safe_load(archivo.read_text(encoding="utf-8"))
        if not isinstance(datos, dict):
            raise LedgerError(f"{archivo} no contiene un mapa YAML")
        datos.setdefault("id", archivo.stem)
        datos["_archivo"] = str(archivo)
        registros.append(datos)
    return sorted(registros, key=lambda r: str(r.get("id")))


def cargar_uno(tipo: str, rid: str, *, raiz: str | Path | None = None) -> dict[str, Any] | None:
    archivo = _dir_tipo(tipo, ledger_dir(raiz)) / f"{rid}.yaml"
    if not archivo.is_file():
        return None
    datos = yaml.safe_load(archivo.read_text(encoding="utf-8"))
    datos.setdefault("id", rid)
    datos["_archivo"] = str(archivo)
    return datos


def actualizar_registro(
    tipo: str,
    rid: str,
    cambios: dict[str, Any],
    motivo: str,
    *,
    raiz: str | Path | None = None,
) -> Path:
    """Aplica cambios a un registro y los anota en su `historial`.

    Nada se pierde: el valor anterior de cada campo queda escrito en el propio
    archivo. Un registro no se borra nunca.
    """
    registro = cargar_uno(tipo, rid, raiz=raiz)
    if registro is None:
        raise LedgerError(f"no existe {tipo}/{rid}")
    registro.pop("_archivo", None)
    anotacion = {"el": _ahora(), "motivo": motivo, "cambios": {}}
    for campo, nuevo in cambios.items():
        anotacion["cambios"][campo] = {"antes": registro.get(campo), "despues": nuevo}
        registro[campo] = nuevo
    registro.setdefault("historial", [])
    registro["historial"].append(anotacion)
    problemas = validar(registro)
    if problemas:
        raise LedgerError(f"tras el cambio, {tipo}/{rid} queda inválido: {problemas}")
    destino = _dir_tipo(tipo, ledger_dir(raiz)) / f"{rid}.yaml"
    _volcar(registro, destino)
    return destino


def cerrar_defecto(
    rid: str, resuelto_por: str, *, raiz: str | Path | None = None
) -> Path:
    """Cierra un defecto dejando escrito qué lo resolvió."""
    return actualizar_registro(
        "defect",
        rid,
        {"estado": "cerrado", "cerrado_el": _hoy(), "resuelto_por": resuelto_por},
        motivo="cerrado",
        raiz=raiz,
    )


# --------------------------------------------------------------------------
# Procedencia automática (CONSTITUTION.md §4)
# --------------------------------------------------------------------------

def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        salida = subprocess.run(
            ["git", *args],
            cwd=str(cwd or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if salida.returncode != 0:
        return None
    return salida.stdout.strip()


def _sha256(ruta: Path, tope_bytes: int = 512 * 1024 * 1024) -> str | None:
    """SHA-256 del archivo. `None` si no existe o si pasa del tope."""
    try:
        if not ruta.is_file() or ruta.stat().st_size > tope_bytes:
            return None
        h = hashlib.sha256()
        with ruta.open("rb") as fh:
            for bloque in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloque)
        return h.hexdigest()
    except OSError:
        return None


def capturar_procedencia(
    *,
    semilla: int | None = None,
    inicio: str | None = None,
    checkpoint: str | Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Captura la procedencia de la corrida SIN que nadie la escriba a mano.

    Esto es lo que `CONSTITUTION.md` §4 pide: comando, commit y log. Se captura
    aquí porque un campo que hay que recordar rellenar acaba vacío.
    """
    base = Path(cwd or Path.cwd())
    commit = _git("rev-parse", "HEAD", cwd=base)
    estado_git = _git("status", "--porcelain", cwd=base)
    fin = _ahora()
    duracion = None
    if inicio:
        try:
            t0 = datetime.strptime(inicio, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            t1 = datetime.strptime(fin, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            duracion = round((t1 - t0).total_seconds(), 1)
        except ValueError:
            duracion = None

    proc: dict[str, Any] = {
        "commit": commit,
        "rama": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=base),
        "arbol_sucio": (bool(estado_git) if estado_git is not None else None),
        "git": "sin repositorio" if commit is None else "ok",
        "comando": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else None,
        "cwd": str(base),
        "semilla": semilla,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "inicio": inicio,
        "fin": fin,
        "duracion_s": duracion,
    }
    if checkpoint is not None:
        ruta = Path(checkpoint)
        proc["checkpoint_ruta"] = str(ruta)
        proc["checkpoint_existe"] = ruta.is_file()
        if ruta.is_file():
            proc["checkpoint_bytes"] = ruta.stat().st_size
            proc["checkpoint_sha256"] = _sha256(ruta)
    return proc


def record_run(
    *,
    id: str,
    titulo: str,
    peldano: str,
    script: str,
    estado: str = "completada",
    metricas: dict[str, Any] | None = None,
    task: int | None = None,
    epoca: int | None = None,
    epocas_planeadas: int | None = None,
    n_val: int | None = None,
    fold: int | None = None,
    n_folds: int | None = None,
    checkpoint: str | Path | None = None,
    defectos_conocidos: list[str] | None = None,
    semilla: int | None = None,
    inicio: str | None = None,
    notas: str | None = None,
    raiz: str | Path | None = None,
    sobrescribir: bool = True,
) -> Path:
    """Registra una corrida capturando su procedencia sola.

    Pensado para llamarse al final de un `train_*.py`. `inicio` es el timestamp
    devuelto por `_ahora()` al arrancar (si se omite, no hay duración).
    """
    campos: dict[str, Any] = {
        "id": id,
        "titulo": titulo,
        "peldano": peldano,
        "estado": estado,
        "script": script,
        "task": task,
        "metricas": metricas or {},
        "epoca": epoca,
        "epocas_planeadas": epocas_planeadas,
        "n_val": n_val,
        "fold": fold,
        "n_folds": n_folds,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "defectos_conocidos": defectos_conocidos or [],
        "procedencia": capturar_procedencia(
            semilla=semilla, inicio=inicio, checkpoint=checkpoint
        ),
        "notas": notas,
    }
    return escribir_registro("run", campos, raiz=raiz, sobrescribir=sobrescribir)


def record_run_safe(**kwargs: Any) -> Path | None:
    """`record_run` que NUNCA levanta.

    Un fallo de logging no puede tumbar ocho horas de GPU. Si algo sale mal
    avisa por stderr y devuelve `None`.
    """
    try:
        return record_run(**kwargs)
    except Exception as exc:  # noqa: BLE001 — deliberado: el ledger nunca mata una corrida
        print(f"[ledger] AVISO: no se pudo registrar la corrida: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Consultas: podredumbre, contradicciones, mejor checkpoint
# --------------------------------------------------------------------------

def revisar_pendientes(*, raiz: str | Path | None = None) -> list[dict[str, Any]]:
    """Hechos y decisiones que llevan demasiado sin revisarse.

    No se puede evaluar `revisar_si` (es prosa deliberadamente), así que lo que
    se hace es sacarlo a la superficie junto con su edad para que un humano
    juzgue. Un hecho del entorno con 40 días es sospechoso aunque nadie lo haya
    tocado.
    """
    pendientes: list[dict[str, Any]] = []
    for tipo, campo_fecha in (("fact", "medido_el"), ("decision", "decidido_el")):
        for reg in cargar(tipo, raiz=raiz):
            if reg.get("estado") == "revertida":
                continue
            referencia = reg.get("revisado_el") or reg.get(campo_fecha)
            edad = _dias_desde(referencia)
            vida = reg.get("vida_util_dias") or VIDA_UTIL_POR_DEFECTO
            if edad is not None and edad >= vida:
                pendientes.append({
                    "tipo": tipo,
                    "id": reg["id"],
                    "titulo": reg.get("titulo"),
                    "edad_dias": edad,
                    "vida_util_dias": vida,
                    "revisar_si": reg.get("revisar_si"),
                })
    return sorted(pendientes, key=lambda p: -p["edad_dias"])


def contradicciones(*, raiz: str | Path | None = None) -> list[dict[str, Any]]:
    """Estados imposibles o sospechosos dentro del ledger.

    Todo lo que aquí se detecta es comprobable sin salir de los archivos: una
    referencia rota, dos checkpoints activos para lo mismo, un activo marcado
    como inexistente pero en uso. Nada de esto requiere red ni SSH.
    """
    avisos: list[dict[str, Any]] = []

    def avisar(nivel: str, mensaje: str, quien: str) -> None:
        avisos.append({"nivel": nivel, "mensaje": mensaje, "registro": quien})

    runs = cargar("run", raiz=raiz)
    defects = cargar("defect", raiz=raiz)
    assets = cargar("asset", raiz=raiz)
    facts = cargar("fact", raiz=raiz)
    decisions = cargar("decision", raiz=raiz)

    ids_defect = {d["id"] for d in defects}
    ids_run = {r["id"] for r in runs}
    ids_fact = {f["id"] for f in facts}

    # 1. Esquema: un registro inválido es una contradicción con su propio tipo.
    for reg in [*runs, *defects, *assets, *facts, *decisions]:
        limpio = {k: v for k, v in reg.items() if k != "_archivo"}
        for problema in validar(limpio):
            avisar("error", f"registro inválido: {problema}", f"{reg['tipo']}/{reg['id']}")

    # 2. Referencias colgantes.
    for r in runs:
        for did in r.get("defectos_conocidos") or []:
            if did not in ids_defect:
                avisar("error", f"apunta al defecto inexistente '{did}'", f"run/{r['id']}")
    for d in defects:
        for rid in d.get("afecta") or []:
            if rid not in ids_run and rid not in ids_defect:
                avisar("aviso", f"'afecta' apunta a '{rid}', que no existe", f"defect/{d['id']}")
    for dec in decisions:
        for ev in dec.get("evidencia") or []:
            if ev not in ids_fact and ev not in ids_run:
                avisar("aviso", f"evidencia '{ev}' no existe en el ledger",
                       f"decision/{dec['id']}")
    for a in assets:
        pid = a.get("producido_por")
        if pid and pid not in ids_run:
            avisar("aviso", f"'producido_por' apunta a '{pid}', que no existe",
                   f"asset/{a['id']}")

    # 3. Activos: liveness y unicidad.
    activos_por_clave: dict[tuple[Any, Any], list[str]] = {}
    for a in assets:
        if a.get("clase") != "checkpoint" or not a.get("activo"):
            continue
        clave = (a.get("task"), a.get("componente"))
        activos_por_clave.setdefault(clave, []).append(a["id"])
    for clave, ids in activos_por_clave.items():
        if len(ids) > 1:
            avisar("error",
                   f"{len(ids)} checkpoints activos para task={clave[0]} "
                   f"componente={clave[1]}: {ids}",
                   "assets")

    # Liveness: un aviso por activo como mucho, del más grave al más leve.
    for a in assets:
        edad = _dias_desde(a.get("verificado_el"))
        if a.get("existe") == "no":
            nivel = "error" if a.get("activo") else "aviso"
            sufijo = " y sigue marcado como ACTIVO" if a.get("activo") else ""
            avisar(nivel, f"ya no existe{sufijo}", f"asset/{a['id']}")
        elif not a.get("efimero"):
            continue
        elif edad is None:
            avisar("aviso", "efímero y nunca verificado "
                            f"(existe: {a.get('existe')})", f"asset/{a['id']}")
        elif edad >= 2:
            avisar("aviso", f"efímero, sin verificar desde hace {edad} días",
                   f"asset/{a['id']}")

    # 4. Corridas que dicen estar vivas desde hace demasiado.
    for r in runs:
        if r.get("estado") != "en_curso":
            continue
        inicio = (r.get("procedencia") or {}).get("inicio")
        edad = _dias_desde(inicio[:10] if isinstance(inicio, str) else None)
        if edad is not None and edad >= 1:
            avisar("aviso",
                   f"sigue marcada 'en_curso' desde hace {edad} días; "
                   "confirmar si terminó o murió", f"run/{r['id']}")

    # 5. Defectos críticos abiertos: siempre a la vista.
    for d in defects:
        if d.get("estado") == "abierto" and d.get("severidad") == "critica":
            avisar("error", f"defecto CRÍTICO abierto: {d.get('titulo')}",
                   f"defect/{d['id']}")

    # 6. Decisiones marcadas para revisión.
    for dec in decisions:
        if dec.get("estado") == "revisar":
            avisar("aviso", f"decisión marcada para revisión: {dec.get('titulo')}",
                   f"decision/{dec['id']}")

    orden = {"error": 0, "aviso": 1}
    return sorted(avisos, key=lambda a: (orden.get(a["nivel"], 9), a["registro"]))


def mejor_checkpoint(task: int, *, raiz: str | Path | None = None) -> list[dict[str, Any]]:
    """Qué checkpoint usar AHORA para una task, con su procedencia y sus peros.

    Devuelve una entrada por componente (keypoint, distancia, registracion…),
    porque una task puede necesitar más de un peso.
    """
    assets = [a for a in cargar("asset", raiz=raiz)
              if a.get("clase") == "checkpoint" and a.get("task") == task]
    runs = {r["id"]: r for r in cargar("run", raiz=raiz)}
    defects = {d["id"]: d for d in cargar("defect", raiz=raiz)}

    salida: list[dict[str, Any]] = []
    for a in sorted(assets, key=lambda x: (not x.get("activo"), str(x.get("componente")))):
        run = runs.get(a.get("producido_por") or "")
        abiertos = []
        if run:
            for did in run.get("defectos_conocidos") or []:
                d = defects.get(did)
                if d and d.get("estado") == "abierto":
                    abiertos.append(d)
        salida.append({
            "asset": a,
            "run": run,
            "defectos_abiertos": sorted(
                abiertos, key=lambda d: _ORDEN_SEVERIDAD.get(d.get("severidad"), 9)
            ),
        })
    return salida


# --------------------------------------------------------------------------
# Utilidades de formato para las vistas
# --------------------------------------------------------------------------

def _fmt(valor: Any) -> str:
    if valor is None:
        return "—"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, float):
        if abs(valor) >= 1000:
            return f"{valor:g}"
        # Se recortan ceros a la derecha, pero nunca por debajo de 2 decimales:
        # un AUC de 0.0000 tiene que verse como 0.0000, no como 0.
        texto = f"{valor:.4f}"
        while texto.endswith("0") and len(texto.split(".")[1]) > 2:
            texto = texto[:-1]
        return texto
    return str(valor)


def _metricas_inline(metricas: dict[str, Any] | None) -> str:
    if not metricas:
        return "—"
    return " · ".join(f"`{k}` {_fmt(v)}" for k, v in metricas.items())


def _commit_corto(run: dict[str, Any]) -> str:
    proc = run.get("procedencia") or {}
    commit = proc.get("commit")
    if not commit:
        return "sin git"
    corto = str(commit)[:8]
    return f"`{corto}`" + ("+sucio" if proc.get("arbol_sucio") else "")

def _cabecera(titulo: str, subtitulo: str) -> list[str]:
    return [
        f"# {titulo}",
        "",
        f"> {subtitulo}",
        ">",
        "> **Archivo generado.** La fuente de verdad es `ledger/`; este `.md` se",
        "> reescribe con `python -m fido.ledger render`. Editarlo a mano no sirve",
        "> de nada: el siguiente render se lo lleva.",
        "",
        f"*Generado el {_ahora()}.*",
        "",
    ]


def _icono_severidad(sev: str | None) -> str:
    return {"critica": "🔴", "alta": "🟠", "media": "🟡", "baja": "⚪"}.get(sev or "", "·")


def _marca_severidad(sev: str | None) -> str:
    """Versión ASCII para la consola: la de Windows no siempre digiere emoji."""
    return {"critica": "!!!", "alta": " !!", "media": "  !", "baja": "   "}.get(sev or "", "   ")


# --------------------------------------------------------------------------
# Vistas generadas
# --------------------------------------------------------------------------

def render_now(*, raiz: str | Path | None = None) -> str:
    """`NOW.md`: qué está pasando ahora mismo y qué reclama atención."""
    runs = cargar("run", raiz=raiz)
    defects = cargar("defect", raiz=raiz)
    assets = cargar("asset", raiz=raiz)
    avisos = contradicciones(raiz=raiz)
    pendientes = revisar_pendientes(raiz=raiz)

    L = _cabecera("NOW — dónde estamos ahora mismo",
                  "Estado vivo del proyecto, derivado del ledger.")

    en_curso = [r for r in runs if r.get("estado") == "en_curso"]
    L += ["## Corriendo ahora", ""]
    if en_curso:
        L += ["| peldaño | corrida | script | métricas al último corte |",
              "|---|---|---|---|"]
        for r in en_curso:
            L.append(
                f"| `{r.get('peldano')}` | `{r['id']}` | `{r.get('script')}` | "
                f"{_metricas_inline(r.get('metricas'))} (ép. {_fmt(r.get('epoca'))}"
                f"/{_fmt(r.get('epocas_planeadas'))}) |"
            )
    else:
        L.append("Nada marcado como `en_curso` en el ledger.")
    L.append("")

    L += ["## Qué peso usar hoy", ""]
    L += ["| task | componente | checkpoint | métrica | ¿sigue vivo? |",
          "|---|---|---|---|---|"]
    hay = False
    for task in (1, 2):
        for entrada in mejor_checkpoint(task, raiz=raiz):
            a = entrada["asset"]
            if not a.get("activo"):
                continue
            hay = True
            run = entrada["run"] or {}
            peros = len(entrada["defectos_abiertos"])
            sufijo = f" ⚠️ {peros} defecto(s) abierto(s)" if peros else ""
            verif = a.get("verificado_el")
            liveness = f"{_fmt(a.get('existe'))} " + (
                f"(verif. {verif})" if verif else "(sin verificar nunca)")
            L.append(
                f"| {task} | {_fmt(a.get('componente'))} | `{a.get('ruta')}` | "
                f"{_metricas_inline(run.get('metricas'))}{sufijo} | {liveness} |"
            )
    if not hay:
        L.append("| — | — | ningún checkpoint marcado como activo | — | — |")
    L += ["", "Detalle completo con `python -m fido.ledger best --task N`.", ""]

    abiertos = [d for d in defects if d.get("estado") == "abierto"]
    abiertos.sort(key=lambda d: _ORDEN_SEVERIDAD.get(d.get("severidad"), 9))
    L += [f"## Defectos abiertos ({len(abiertos)})", ""]
    if abiertos:
        L += ["| | severidad | defecto | dónde |", "|---|---|---|---|"]
        for d in abiertos:
            L.append(
                f"| {_icono_severidad(d.get('severidad'))} | {_fmt(d.get('severidad'))} | "
                f"{d.get('titulo')} | `{_fmt(d.get('donde'))}` |"
            )
    else:
        L.append("Ninguno. (Improbable: revisa si el inventario está al día.)")
    L.append("")

    errores = [a for a in avisos if a["nivel"] == "error"]
    otros = [a for a in avisos if a["nivel"] != "error"]
    L += ["## Alertas del ledger", ""]
    def _lineas(grupo: list[dict[str, Any]]) -> list[str]:
        # Un mismo aviso repetido en ocho activos se lee mejor en una línea.
        por_mensaje: dict[str, list[str]] = {}
        for a in grupo:
            por_mensaje.setdefault(a["mensaje"], []).append(a["registro"])
        salida = []
        for mensaje, quienes in por_mensaje.items():
            if len(quienes) == 1:
                salida.append(f"- `{quienes[0]}` — {mensaje}")
            else:
                salida.append(f"- **{len(quienes)} registros** — {mensaje}: "
                              + ", ".join(f"`{q}`" for q in quienes))
        return salida

    if errores:
        L += ["**Errores**", ""] + _lineas(errores) + [""]
    if otros:
        L += ["**Avisos**", ""] + _lineas(otros) + [""]
    if not avisos:
        L += ["Ninguna. El ledger es internamente consistente.", ""]

    L += ["## Por revisar (podredumbre)", ""]
    if pendientes:
        L += ["| registro | edad | revisar si… |", "|---|---|---|"]
        for p in pendientes:
            L.append(
                f"| `{p['tipo']}/{p['id']}` — {p['titulo']} | {p['edad_dias']} d "
                f"(vida útil {p['vida_util_dias']}) | {p['revisar_si']} |"
            )
    else:
        L.append("Nada ha cumplido su vida útil todavía.")
    L.append("")

    efimeros = [a for a in assets if a.get("efimero")]
    L += ["## Riesgo de pérdida", "",
          f"{len(efimeros)} de {len(assets)} activos viven en almacenamiento efímero "
          "(el pod). Si el pod muere, se pierden.", ""]
    return "\n".join(L).rstrip() + "\n"


def render_results(*, raiz: str | Path | None = None) -> str:
    """`RESULTS.md`: un renglón por corrida, con su procedencia al lado."""
    runs = cargar("run", raiz=raiz)
    facts = cargar("fact", raiz=raiz)
    defects = {d["id"]: d for d in cargar("defect", raiz=raiz)}

    L = _cabecera("RESULTS — qué funcionó, qué no, y con qué evidencia",
                  "Un renglón por corrida con número. Los fracasos cuentan igual "
                  "que los éxitos (`CONSTITUTION.md` §5).")

    L += ["## Corridas", ""]
    L += ["| peldaño | corrida | estado | métricas | ép. | n val | fold | commit |",
          "|---|---|---|---|---|---|---|---|"]
    icono = {"completada": "✅", "en_curso": "🔄", "murio": "💥", "abortada": "🚫"}
    for r in sorted(runs, key=lambda r: str(r.get("peldano"))):
        L.append(
            f"| `{r.get('peldano')}` | `{r['id']}` | "
            f"{icono.get(r.get('estado'), '')} {r.get('estado')} | "
            f"{_metricas_inline(r.get('metricas'))} | {_fmt(r.get('epoca'))} | "
            f"{_fmt(r.get('n_val'))} | {_fmt(r.get('fold'))}/{_fmt(r.get('n_folds'))} | "
            f"{_commit_corto(r)} |"
        )
    L.append("")

    L += ["## Cómo leer cada número", "",
          "Los avisos de abajo no son opcionales: un número con un defecto abierto",
          "encima no significa lo que parece.", ""]
    for r in sorted(runs, key=lambda r: str(r.get("peldano"))):
        proc = r.get("procedencia") or {}
        L += [f"### `{r['id']}` — {r.get('titulo')}", ""]
        L += [f"- **Peldaño**: `{r.get('peldano')}` · **estado**: {r.get('estado')}"]
        L += [f"- **Métricas**: {_metricas_inline(r.get('metricas'))}"]
        L += [f"- **Comando**: `{_fmt(proc.get('comando'))}`"]
        L += [f"- **Commit**: {_commit_corto(r)} · **semilla**: {_fmt(proc.get('semilla'))}"
              f" · **host**: `{_fmt(proc.get('hostname'))}`"]
        if proc.get("inicio") and proc.get("fin"):
            dur = f" ({proc['duracion_s']} s)" if proc.get("duracion_s") else ""
            L += [f"- **Cuándo**: {proc['inicio']} → {proc['fin']}{dur}"]
        elif proc.get("inicio"):
            L += [f"- **Cuándo**: desde {proc['inicio']}"]
        if r.get("checkpoint"):
            L += [f"- **Checkpoint**: `{r['checkpoint']}`"]
        for did in r.get("defectos_conocidos") or []:
            d = defects.get(did)
            if not d:
                continue
            estado = "ABIERTO" if d.get("estado") == "abierto" else "cerrado"
            L += [f"- {_icono_severidad(d.get('severidad'))} **{estado}** "
                  f"(`{did}`): {d.get('titulo')}"]
        if r.get("notas"):
            L += ["", f"{r['notas']}"]
        L.append("")

    cotas = [f for f in facts if f.get("ambito") == "metrica"]
    if cotas:
        L += ["## Cotas y relaciones medidas", "",
              "Medidos, no estimados. Contexto para juzgar cualquier resultado futuro.", ""]
        L += ["| cota o relación | valor | de dónde sale |", "|---|---|---|"]
        for f in cotas:
            unidad = f" {f['unidad']}" if f.get("unidad") else ""
            L.append(f"| {f.get('titulo')} | **{_fmt(f.get('valor'))}**{unidad} | "
                     f"`{_fmt(f.get('medido_con'))}` |")
        L.append("")

    comp = [f for f in facts if f.get("ambito") == "competencia"]
    if comp:
        L += ["## Contra qué se compite", ""]
        for f in comp:
            L += [f"- **{f.get('titulo')}**: {_fmt(f.get('valor'))} "
                  f"*(medido el {f.get('medido_el')})*"]
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def render_the_map(*, raiz: str | Path | None = None) -> str:
    """`THE_MAP.md`: qué existe, dónde vive y con qué fecha se midió."""
    facts = cargar("fact", raiz=raiz)
    assets = cargar("asset", raiz=raiz)
    decisions = cargar("decision", raiz=raiz)
    runs = {r["id"]: r for r in cargar("run", raiz=raiz)}

    L = _cabecera("EL MAPA — qué existe y en qué estado",
                  "Hechos medidos, activos y decisiones vigentes.")

    for ambito, titulo in (("entorno", "Entorno — medido, no recordado"),
                           ("datos", "Los datos")):
        subset = [f for f in facts if f.get("ambito") == ambito]
        if not subset:
            continue
        L += [f"## {titulo}", ""]
        L += ["| hecho | valor | medido | revisar si… |", "|---|---|---|---|"]
        for f in subset:
            unidad = f" {f['unidad']}" if f.get("unidad") else ""
            L.append(f"| {f.get('titulo')} | **{_fmt(f.get('valor'))}**{unidad} | "
                     f"{f.get('medido_el')} | {_fmt(f.get('revisar_si'))} |")
        L.append("")
        con_notas = [f for f in subset if f.get("notas")]
        if con_notas:
            L += [f"- **{f.get('titulo')}**: {f['notas']}" for f in con_notas]
            L.append("")

    L += ["## Activos — checkpoints", ""]
    ck = [a for a in assets if a.get("clase") == "checkpoint"]
    L += ["| | task | componente | ruta | producido por | ¿vive? |",
          "|---|---|---|---|---|---|"]
    for a in ck:
        marca = "★" if a.get("activo") else ""
        origen = a.get("producido_por")
        run = runs.get(origen or "")
        origen_txt = f"`{origen}`" if origen else "—"
        if run:
            origen_txt += f" ({_metricas_inline(run.get('metricas'))})"
        L.append(f"| {marca} | {_fmt(a.get('task'))} | {_fmt(a.get('componente'))} | "
                 f"`{a.get('ruta')}` | {origen_txt} | {_fmt(a.get('existe'))} |")
    L += ["", "★ = el que se usa ahora mismo para esa task+componente.", ""]

    L += ["## Activos — caches", "",
          "Sin estos, un entrenamiento pasa de minutos a horas. No son opcionales.", ""]
    L += ["| cache | ruta | host | ¿vive? |", "|---|---|---|---|"]
    for a in [a for a in assets if a.get("clase") == "cache"]:
        L.append(f"| {a.get('titulo')} | `{a.get('ruta')}` | `{a.get('host')}` | "
                 f"{_fmt(a.get('existe'))} |")
    L.append("")

    efimeros = [a for a in assets if a.get("efimero")]
    if efimeros:
        L += ["> ⚠️ **Todo lo marcado como efímero vive solo en el pod.** "
              f"Son {len(efimeros)} activos. Si el pod desaparece, hay que "
              "reconstruirlos o volver a entrenar.", ""]

    L += ["## Decisiones", ""]
    for estado, titulo in (("vigente", "Vigentes"), ("revisar", "En revisión"),
                           ("revertida", "Revertidas")):
        subset = [d for d in decisions if d.get("estado") == estado]
        if not subset:
            continue
        L += [f"### {titulo}", ""]
        for d in subset:
            L += [f"**{d.get('titulo')}** *(desde {d.get('decidido_el')})*", "",
                  f"- Qué: {d.get('decision')}",
                  f"- Por qué: {d.get('porque')}",
                  f"- Se revierte si: {d.get('revisar_si')}", ""]
    return "\n".join(L).rstrip() + "\n"


def render_todo(*, raiz: str | Path | None = None, destino: Path | None = None) -> list[Path]:
    """Escribe las tres vistas en `ledger/_rendered/`."""
    carpeta = Path(destino) if destino else (ledger_dir(raiz) / "_rendered")
    carpeta.mkdir(parents=True, exist_ok=True)
    escritos = []
    for nombre, fn in (("NOW.md", render_now), ("RESULTS.md", render_results),
                       ("THE_MAP.md", render_the_map)):
        ruta = carpeta / nombre
        ruta.write_text(fn(raiz=raiz), encoding="utf-8")
        escritos.append(ruta)
    return escritos


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cmd_status(args: argparse.Namespace) -> int:
    raiz = args.ledger_dir
    runs = cargar("run", raiz=raiz)
    defects = cargar("defect", raiz=raiz)
    facts = cargar("fact", raiz=raiz)
    decisions = cargar("decision", raiz=raiz)
    assets = cargar("asset", raiz=raiz)

    print(f"Ledger: {ledger_dir(raiz)}")
    print(f"  {len(runs)} corridas · {len(defects)} defectos · {len(facts)} hechos · "
          f"{len(decisions)} decisiones · {len(assets)} activos")
    print()
    en_curso = [r for r in runs if r.get("estado") == "en_curso"]
    print(f"Corriendo ahora ({len(en_curso)}):")
    for r in en_curso or []:
        print(f"  >> {r['id']:<34} {r.get('peldano')}  "
              f"{_metricas_inline(r.get('metricas'))}")
    if not en_curso:
        print("  (nada)")
    print()
    abiertos = sorted([d for d in defects if d.get("estado") == "abierto"],
                      key=lambda d: _ORDEN_SEVERIDAD.get(d.get("severidad"), 9))
    print(f"Defectos abiertos ({len(abiertos)}):")
    for d in abiertos:
        print(f"  {_marca_severidad(d.get('severidad'))} {d.get('severidad'):<8} "
              f"{d['id']:<34} {d.get('titulo')}")
    print()
    print("Checkpoints activos:")
    algo = False
    for a in assets:
        if a.get("clase") == "checkpoint" and a.get("activo"):
            algo = True
            print(f"  *  task {_fmt(a.get('task'))} / {_fmt(a.get('componente')):<14} "
                  f"{a.get('ruta')}  [existe: {a.get('existe')}]")
    if not algo:
        print("  (ninguno)")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    raiz = args.ledger_dir
    avisos = contradicciones(raiz=raiz)
    pendientes = revisar_pendientes(raiz=raiz)

    errores = [a for a in avisos if a["nivel"] == "error"]
    print(f"Contradicciones: {len(errores)} error(es), "
          f"{len(avisos) - len(errores)} aviso(s)")
    for a in avisos:
        marca = "ERROR" if a["nivel"] == "error" else "aviso"
        print(f"  [{marca}] {a['registro']}: {a['mensaje']}")
    if not avisos:
        print("  (ninguna)")
    print()
    print(f"Por revisar — cumplieron su vida útil ({len(pendientes)}):")
    for p in pendientes:
        print(f"  {p['tipo']}/{p['id']}  ({p['edad_dias']}d ≥ {p['vida_util_dias']}d)")
        print(f"      revisar si: {p['revisar_si']}")
    if not pendientes:
        print("  (ninguno)")
    return 1 if errores else 0


def _cmd_best(args: argparse.Namespace) -> int:
    raiz = args.ledger_dir
    entradas = mejor_checkpoint(args.task, raiz=raiz)
    if not entradas:
        print(f"No hay checkpoints registrados para task {args.task}.")
        return 1
    print(f"Task {args.task} — qué peso usar ahora mismo\n")
    for e in entradas:
        a, run = e["asset"], e["run"]
        marca = "* ACTIVO     " if a.get("activo") else "  (no activo)"
        print(f"{marca}  componente: {_fmt(a.get('componente'))}")
        print(f"    ruta      : {a.get('ruta')}")
        print(f"    host      : {a.get('host')}  "
              f"(efímero: {_fmt(a.get('efimero'))}, existe: {a.get('existe')}, "
              f"verificado: {_fmt(a.get('verificado_el'))})")
        if run:
            proc = run.get("procedencia") or {}
            print(f"    corrida   : {run['id']}  [{run.get('estado')}] "
                  f"peldaño {run.get('peldano')}")
            print(f"    métricas  : {_metricas_inline(run.get('metricas'))} "
                  f"(época {_fmt(run.get('epoca'))}, n_val {_fmt(run.get('n_val'))})")
            print(f"    comando   : {_fmt(proc.get('comando'))}")
            print(f"    commit    : {_fmt(proc.get('commit'))}"
                  f"{' (árbol sucio)' if proc.get('arbol_sucio') else ''}"
                  f"  semilla: {_fmt(proc.get('semilla'))}")
        else:
            print("    corrida   : [!] HUERFANO - ninguna corrida registrada lo produjo")
        if e["defectos_abiertos"]:
            print(f"    [!] {len(e['defectos_abiertos'])} defecto(s) abierto(s) "
                  "que afectan la lectura de estos números:")
            for d in e["defectos_abiertos"]:
                print(f"        {_marca_severidad(d.get('severidad'))} "
                      f"[{d.get('severidad')}] {d.get('titulo')}")
        print()
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    escritos = render_todo(raiz=args.ledger_dir,
                           destino=Path(args.out) if args.out else None)
    for ruta in escritos:
        print(f"escrito: {ruta}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fido.ledger",
        description="Ledger del proyecto FIDO: registros YAML como fuente de verdad.",
    )
    parser.add_argument("--ledger-dir", default=None,
                        help="raíz del ledger (por defecto: la del repo o $FIDO_LEDGER_DIR)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="qué hay ahora: corridas, defectos, pesos activos")
    sub.add_parser("check", help="podredumbre y contradicciones")

    p_best = sub.add_parser("best", help="qué checkpoint usar para una task")
    p_best.add_argument("--task", type=int, required=True, choices=(1, 2))

    p_render = sub.add_parser("render", help="regenerar las vistas .md")
    p_render.add_argument("--out", default=None,
                          help="carpeta destino (por defecto ledger/_rendered/)")

    for flujo in (sys.stdout, sys.stderr):
        try:  # la consola de Windows suele venir en cp1252 y ahoga los acentos
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    args = parser.parse_args(list(argv) if argv is not None else None)
    return {
        "status": _cmd_status,
        "check": _cmd_check,
        "best": _cmd_best,
        "render": _cmd_render,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
