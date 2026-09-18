"""Tests del ledger.

Cada test usa un ledger propio en `tmp_path` — nunca toca `ledger/` del repo.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from fido import ledger as L


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _fecha_hace(dias: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=dias)).strftime("%Y-%m-%d")


@pytest.fixture()
def raiz(tmp_path: Path) -> Path:
    for plural in L.TIPOS.values():
        (tmp_path / plural).mkdir(parents=True)
    return tmp_path


def _defecto(raiz: Path, rid="def-x", estado="abierto", severidad="alta", **kw) -> None:
    L.escribir_registro("defect", {
        "id": rid, "titulo": "un defecto", "estado": estado, "severidad": severidad,
        "sintoma": "algo pasa", "abierto_el": _fecha_hace(1), **kw,
    }, raiz=raiz)


def _run(raiz: Path, rid="run-x", **kw) -> None:
    campos = {
        "id": rid, "titulo": "una corrida", "peldano": "10-x", "estado": "completada",
        "script": "fido.train.x", "procedencia": {"comando": "x", "commit": None},
    }
    campos.update(kw)
    L.escribir_registro("run", campos, raiz=raiz)


def _asset(raiz: Path, rid="ck-x", **kw) -> None:
    campos = {
        "id": rid, "titulo": "un peso", "clase": "checkpoint", "ruta": "/w/model_0.pth",
        "host": "pod:abc", "efimero": True, "existe": "si", "task": 1,
        "componente": "distancia", "activo": True, "verificado_el": _fecha_hace(0),
    }
    campos.update(kw)
    L.escribir_registro("asset", campos, raiz=raiz)


# --------------------------------------------------------------------------
# Esquema y E/S
# --------------------------------------------------------------------------

def test_escribir_y_releer_conserva_los_campos(raiz: Path) -> None:
    _run(raiz, rid="run-01", metricas={"val_auc": 0.5681}, epoca=2, task=1)
    leido = L.cargar_uno("run", "run-01", raiz=raiz)
    assert leido is not None
    assert leido["metricas"]["val_auc"] == pytest.approx(0.5681)
    assert leido["epoca"] == 2
    assert leido["tipo"] == "run"


def test_el_archivo_es_yaml_legible_a_ojo(raiz: Path) -> None:
    """Si el módulo se rompe, el ledger debe seguir leyéndose con `cat`."""
    _run(raiz, rid="run-01", notas="una nota con acentos: región")
    texto = (raiz / "runs" / "run-01.yaml").read_text(encoding="utf-8")
    assert "región" in texto            # sin escapes \uXXXX
    assert texto.startswith("id: run-01")  # el id manda, orden canónico
    assert yaml.safe_load(texto)["titulo"] == "una corrida"


def test_no_se_pisa_un_registro_existente(raiz: Path) -> None:
    _run(raiz, rid="run-01")
    with pytest.raises(L.LedgerError, match="ya existe"):
        _run(raiz, rid="run-01")


def test_rechaza_campo_obligatorio_ausente(raiz: Path) -> None:
    with pytest.raises(L.LedgerError, match="obligatorio"):
        L.escribir_registro("defect", {
            "id": "d1", "titulo": "t", "estado": "abierto", "severidad": "alta",
            "abierto_el": _fecha_hace(0),  # falta 'sintoma'
        }, raiz=raiz)


def test_rechaza_valor_fuera_del_enum(raiz: Path) -> None:
    with pytest.raises(L.LedgerError, match="severidad"):
        _defecto(raiz, severidad="catastrofica")


def test_rechaza_campo_inventado(raiz: Path) -> None:
    with pytest.raises(L.LedgerError, match="fuera del esquema"):
        _defecto(raiz, prioridad="urgente")


def test_rechaza_fecha_mal_formada(raiz: Path) -> None:
    with pytest.raises(L.LedgerError, match="YYYY-MM-DD"):
        _defecto(raiz, abierto_el="17/08/2026")


def test_actualizar_deja_rastro_del_valor_anterior(raiz: Path) -> None:
    _defecto(raiz, rid="def-1")
    L.cerrar_defecto("def-1", "lo arregló el peldaño 40", raiz=raiz)
    leido = L.cargar_uno("defect", "def-1", raiz=raiz)
    assert leido["estado"] == "cerrado"
    assert leido["resuelto_por"].startswith("lo arregló")
    hist = leido["historial"][-1]
    assert hist["cambios"]["estado"] == {"antes": "abierto", "despues": "cerrado"}


def test_actualizar_registro_inexistente_falla(raiz: Path) -> None:
    with pytest.raises(L.LedgerError, match="no existe"):
        L.actualizar_registro("defect", "fantasma", {"estado": "cerrado"}, "x", raiz=raiz)


# --------------------------------------------------------------------------
# Procedencia automática
# --------------------------------------------------------------------------

def test_procedencia_captura_lo_que_nadie_escribiria(raiz: Path) -> None:
    proc = L.capturar_procedencia(semilla=7, inicio=L._ahora())
    for campo in ("comando", "cwd", "hostname", "python", "plataforma", "inicio",
                  "fin", "duracion_s", "semilla", "commit", "arbol_sucio"):
        assert campo in proc, campo
    assert proc["semilla"] == 7
    assert proc["duracion_s"] is not None and proc["duracion_s"] >= 0


def test_procedencia_sobrevive_a_no_haber_repo_git(tmp_path: Path) -> None:
    """El repo aún no está bajo git; eso no puede romper el registro."""
    proc = L.capturar_procedencia(cwd=tmp_path)
    assert proc["commit"] is None
    assert proc["git"] == "sin repositorio"


def test_procedencia_hashea_el_checkpoint(raiz: Path, tmp_path: Path) -> None:
    peso = tmp_path / "model_0.pth"
    peso.write_bytes(b"pesos falsos")
    proc = L.capturar_procedencia(checkpoint=peso)
    assert proc["checkpoint_existe"] is True
    assert proc["checkpoint_bytes"] == len(b"pesos falsos")
    assert len(proc["checkpoint_sha256"]) == 64


def test_procedencia_con_checkpoint_ausente_no_revienta(raiz: Path, tmp_path: Path) -> None:
    proc = L.capturar_procedencia(checkpoint=tmp_path / "no-existe.pth")
    assert proc["checkpoint_existe"] is False
    assert "checkpoint_sha256" not in proc


def test_record_run_captura_procedencia_sin_pedirsela_a_nadie(raiz: Path) -> None:
    L.record_run(id="run-auto", titulo="t", peldano="10-x",
                 script="fido.train.x", metricas={"auc": 0.1}, semilla=3, raiz=raiz)
    leido = L.cargar_uno("run", "run-auto", raiz=raiz)
    assert leido["procedencia"]["semilla"] == 3
    assert leido["procedencia"]["hostname"]
    assert leido["procedencia"]["comando"]


def test_record_run_safe_nunca_tumba_un_entrenamiento(raiz: Path, capsys) -> None:
    """Perder 8 h de GPU por un error de logging sería inaceptable."""
    salida = L.record_run_safe(id="ID INVÁLIDO", titulo="t", peldano="p",
                               script="s", raiz=raiz)
    assert salida is None
    assert "AVISO" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Podredumbre
# --------------------------------------------------------------------------

def test_un_hecho_viejo_sale_por_revisar(raiz: Path) -> None:
    L.escribir_registro("fact", {
        "id": "f-viejo", "titulo": "RAM del pod", "ambito": "entorno", "valor": 187,
        "unidad": "GB", "medido_el": _fecha_hace(40), "medido_con": "free -g",
        "revisar_si": "el pod cambia", "vida_util_dias": 7,
    }, raiz=raiz)
    pendientes = L.revisar_pendientes(raiz=raiz)
    assert [p["id"] for p in pendientes] == ["f-viejo"]
    assert pendientes[0]["revisar_si"] == "el pod cambia"


def test_un_hecho_fresco_no_molesta(raiz: Path) -> None:
    L.escribir_registro("fact", {
        "id": "f-nuevo", "titulo": "RAM", "ambito": "entorno", "valor": 187,
        "medido_el": _fecha_hace(1), "revisar_si": "el pod cambia",
        "vida_util_dias": 30,
    }, raiz=raiz)
    assert L.revisar_pendientes(raiz=raiz) == []


def test_revisar_reinicia_el_reloj(raiz: Path) -> None:
    L.escribir_registro("decision", {
        "id": "d-1", "titulo": "Task 2 primero", "estado": "vigente",
        "decision": "atacar T2", "porque": "campo débil", "revisar_si": "T1 se atrasa",
        "decidido_el": _fecha_hace(60), "vida_util_dias": 14,
    }, raiz=raiz)
    assert len(L.revisar_pendientes(raiz=raiz)) == 1
    L.actualizar_registro("decision", "d-1", {"revisado_el": _fecha_hace(0)},
                          "revisada hoy", raiz=raiz)
    assert L.revisar_pendientes(raiz=raiz) == []


# --------------------------------------------------------------------------
# Contradicciones
# --------------------------------------------------------------------------

def test_detecta_referencia_a_defecto_inexistente(raiz: Path) -> None:
    _run(raiz, rid="run-1", defectos_conocidos=["def-fantasma"])
    avisos = L.contradicciones(raiz=raiz)
    assert any("def-fantasma" in a["mensaje"] and a["nivel"] == "error" for a in avisos)


def test_detecta_checkpoint_activo_que_ya_no_existe(raiz: Path) -> None:
    _asset(raiz, rid="ck-muerto", existe="no", activo=True)
    avisos = L.contradicciones(raiz=raiz)
    assert any("ya no existe" in a["mensaje"] for a in avisos)


def test_detecta_dos_activos_para_la_misma_task_y_componente(raiz: Path) -> None:
    _asset(raiz, rid="ck-a")
    _asset(raiz, rid="ck-b")
    avisos = L.contradicciones(raiz=raiz)
    assert any("checkpoints activos" in a["mensaje"] for a in avisos)


def test_detecta_activo_efimero_sin_verificar(raiz: Path) -> None:
    _asset(raiz, rid="ck-viejo", verificado_el=_fecha_hace(9))
    avisos = L.contradicciones(raiz=raiz)
    assert any("sin verificar desde hace 9 días" in a["mensaje"] for a in avisos)


def test_detecta_corrida_en_curso_desde_hace_dias(raiz: Path) -> None:
    _run(raiz, rid="run-zombi", estado="en_curso",
         procedencia={"comando": "x", "inicio": f"{_fecha_hace(3)}T00:00:00Z"})
    avisos = L.contradicciones(raiz=raiz)
    assert any("en_curso" in a["mensaje"] for a in avisos)


def test_un_defecto_critico_abierto_siempre_se_ve(raiz: Path) -> None:
    _defecto(raiz, rid="def-critico", severidad="critica")
    avisos = L.contradicciones(raiz=raiz)
    assert any(a["nivel"] == "error" and "CRÍTICO" in a["mensaje"] for a in avisos)


def test_un_ledger_sano_no_inventa_avisos(raiz: Path) -> None:
    _run(raiz, rid="run-1")
    _asset(raiz, rid="ck-1", producido_por="run-1")
    _defecto(raiz, rid="def-1", estado="cerrado", cerrado_el=_fecha_hace(0),
             severidad="media")
    assert L.contradicciones(raiz=raiz) == []


def test_detecta_registro_invalido_escrito_a_mano(raiz: Path) -> None:
    """El ledger es editable con un editor de texto; `check` es la red de seguridad."""
    (raiz / "defects" / "a-mano.yaml").write_text(
        "tipo: defect\nid: a-mano\ntitulo: t\nestado: abierto\nseveridad: MUY_ALTA\n"
        "sintoma: s\nabierto_el: 2026-08-17\n", encoding="utf-8")
    avisos = L.contradicciones(raiz=raiz)
    assert any("inválido" in a["mensaje"] for a in avisos)


# --------------------------------------------------------------------------
# best
# --------------------------------------------------------------------------

def test_best_devuelve_el_activo_con_su_procedencia_y_sus_peros(raiz: Path) -> None:
    _defecto(raiz, rid="def-inflado", severidad="alta")
    _run(raiz, rid="run-1", task=1, metricas={"val_distance_auc": 0.5681},
         defectos_conocidos=["def-inflado"])
    _asset(raiz, rid="ck-1", producido_por="run-1")
    (entrada,) = L.mejor_checkpoint(1, raiz=raiz)
    assert entrada["asset"]["ruta"] == "/w/model_0.pth"
    assert entrada["run"]["metricas"]["val_distance_auc"] == pytest.approx(0.5681)
    assert [d["id"] for d in entrada["defectos_abiertos"]] == ["def-inflado"]


def test_best_no_esconde_un_checkpoint_huerfano(raiz: Path) -> None:
    _asset(raiz, rid="ck-huerfano", producido_por=None)
    (entrada,) = L.mejor_checkpoint(1, raiz=raiz)
    assert entrada["run"] is None


def test_best_ignora_los_defectos_ya_cerrados(raiz: Path) -> None:
    _defecto(raiz, rid="def-cerrado", estado="cerrado", cerrado_el=_fecha_hace(0))
    _run(raiz, rid="run-1", task=1, defectos_conocidos=["def-cerrado"])
    _asset(raiz, rid="ck-1", producido_por="run-1")
    (entrada,) = L.mejor_checkpoint(1, raiz=raiz)
    assert entrada["defectos_abiertos"] == []


def test_best_pone_primero_el_activo(raiz: Path) -> None:
    _asset(raiz, rid="ck-viejo", activo=False, componente="distancia")
    _asset(raiz, rid="ck-nuevo", activo=True, componente="distancia")
    ids = [e["asset"]["id"] for e in L.mejor_checkpoint(1, raiz=raiz)]
    assert ids[0] == "ck-nuevo"


# --------------------------------------------------------------------------
# Vistas
# --------------------------------------------------------------------------

def _ledger_de_muestra(raiz: Path) -> None:
    _defecto(raiz, rid="def-inflado", severidad="alta", donde="train_task1_unet.py")
    _run(raiz, rid="run-1", task=1, peldano="10-t1-distancia-unet",
         metricas={"val_distance_auc": 0.5681}, epoca=2, n_val=12566,
         fold=0, n_folds=5, defectos_conocidos=["def-inflado"],
         procedencia={"comando": "train_task1_unet.py --epochs 15", "commit": "abc123def456",
                      "arbol_sucio": False, "semilla": 0, "hostname": "pod"})
    _run(raiz, rid="run-2", task=2, estado="en_curso", peldano="50-t1-keypoint")
    _asset(raiz, rid="ck-1", producido_por="run-1")
    _asset(raiz, rid="cache-1", clase="cache", titulo="en-face Task 2",
           ruta="/root/data_cache/Task2_enface", activo=False, componente=None, task=None)
    L.escribir_registro("fact", {
        "id": "f-piso", "titulo": "Piso trivial de distance_auc", "ambito": "metrica",
        "valor": 0.0445, "medido_el": _fecha_hace(2), "medido_con": "explore_task1_gt.py",
        "revisar_si": "cambia el GT", "vida_util_dias": 365,
    }, raiz=raiz)
    L.escribir_registro("decision", {
        "id": "dec-1", "titulo": "Dos modelos, un zip", "estado": "vigente",
        "decision": "model_0.pth y model_1.pth", "porque": "el formato lo permite",
        "revisar_si": "el contrato de entrega cambia", "decidido_el": _fecha_hace(3),
    }, raiz=raiz)


def test_las_tres_vistas_se_escriben(raiz: Path) -> None:
    _ledger_de_muestra(raiz)
    escritos = L.render_todo(raiz=raiz)
    assert {p.name for p in escritos} == {"NOW.md", "RESULTS.md", "THE_MAP.md"}
    assert all(p.read_text(encoding="utf-8").strip() for p in escritos)


def test_render_no_toca_los_md_de_la_raiz_del_repo(raiz: Path) -> None:
    """Los .md de la raíz están en uso; el render escribe en _rendered/."""
    _ledger_de_muestra(raiz)
    (raiz / "NOW.md").write_text("NO ME TOQUES", encoding="utf-8")
    L.render_todo(raiz=raiz)
    assert (raiz / "NOW.md").read_text(encoding="utf-8") == "NO ME TOQUES"
    assert (raiz / "_rendered" / "NOW.md").exists()


def test_now_muestra_lo_que_corre_y_lo_que_duele(raiz: Path) -> None:
    _ledger_de_muestra(raiz)
    now = L.render_now(raiz=raiz)
    assert "run-2" in now                    # la corrida en curso
    assert "0.5681" in now                   # la métrica del peso activo
    assert "un defecto" in now               # el defecto abierto
    assert "Archivo generado" in now         # el aviso de no editar a mano


def test_results_lleva_comando_y_commit_de_cada_numero(raiz: Path) -> None:
    """CONSTITUTION.md §4: un número sin su comando y su commit es una anécdota."""
    _ledger_de_muestra(raiz)
    res = L.render_results(raiz=raiz)
    assert "0.5681" in res
    assert "train_task1_unet.py --epochs 15" in res
    assert "abc123de" in res
    assert "ABIERTO" in res                  # el defecto que infla ese número


def test_the_map_marca_los_activos_efimeros(raiz: Path) -> None:
    _ledger_de_muestra(raiz)
    mapa = L.render_the_map(raiz=raiz)
    assert "efímero" in mapa
    assert "/root/data_cache/Task2_enface" in mapa
    assert "Dos modelos, un zip" in mapa


def test_las_tablas_generadas_no_se_rompen_por_celdas_vacias(raiz: Path) -> None:
    _run(raiz, rid="run-pelado")
    for texto in (L.render_now(raiz=raiz), L.render_results(raiz=raiz),
                  L.render_the_map(raiz=raiz)):
        for linea in texto.splitlines():
            if linea.startswith("|"):
                assert linea.endswith("|"), linea
                assert "||" not in linea, linea


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_status_y_render(raiz: Path, capsys) -> None:
    _ledger_de_muestra(raiz)
    assert L.main(["--ledger-dir", str(raiz), "status"]) == 0
    salida = capsys.readouterr().out
    assert "Defectos abiertos (1)" in salida
    assert L.main(["--ledger-dir", str(raiz), "render"]) == 0
    assert (raiz / "_rendered" / "RESULTS.md").exists()


def test_cli_check_sale_con_1_si_hay_errores(raiz: Path, capsys) -> None:
    _run(raiz, rid="run-1", defectos_conocidos=["fantasma"])
    assert L.main(["--ledger-dir", str(raiz), "check"]) == 1
    assert "ERROR" in capsys.readouterr().out


def test_cli_check_sale_con_0_si_todo_esta_sano(raiz: Path) -> None:
    _run(raiz, rid="run-1")
    assert L.main(["--ledger-dir", str(raiz), "check"]) == 0


def test_cli_best(raiz: Path, capsys) -> None:
    _ledger_de_muestra(raiz)
    assert L.main(["--ledger-dir", str(raiz), "best", "--task", "1"]) == 0
    salida = capsys.readouterr().out
    assert "* ACTIVO" in salida
    assert "0.5681" in salida
    assert "defecto(s) abierto(s)" in salida


def test_cli_invocable_como_modulo(raiz: Path) -> None:
    """`python -m fido.ledger` tiene que funcionar de verdad, no solo main()."""
    _run(raiz, rid="run-1")
    src = str(Path(L.__file__).resolve().parents[1])
    proc = subprocess.run(
        [sys.executable, "-m", "fido.ledger", "--ledger-dir", str(raiz), "status"],
        capture_output=True, text=True, cwd=src, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "1 corridas" in proc.stdout
