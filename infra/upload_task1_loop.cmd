@echo off
REM Sube Task 1 al volumen de RunPod, reintentando hasta lograrlo.
REM
REM Corre desacoplado: lanzalo con
REM     powershell -Command "Start-Process -FilePath 'infra\upload_task1_loop.cmd' -WindowStyle Hidden"
REM y sigue el avance con
REM     type logs\task1_upload.log
REM
REM Cada pasada de resumable_upload.py retoma donde quedo la anterior, asi que
REM el bucle siempre avanza: nunca reempieza un archivo desde cero.

cd /d "%~dp0.."
if not exist logs mkdir logs

set ATTEMPT=0

:retry
set /a ATTEMPT+=1
echo. >> logs\task1_upload.log
echo ===== PASADA %ATTEMPT% - %DATE% %TIME% ===== >> logs\task1_upload.log

python infra\resumable_upload.py ^
  "data\Task 1\Scenario_01.zip" ^
  "data\Task 1\Scenario_02.zip" ^
  "data\Task 1\Scenario_03.zip" ^
  "data\Task 1\Scenario_04.zip" ^
  "data\Task 1\Scenario_05.zip" ^
  "data\Task 1\Scenario_06.zip" ^
  "data\Task 1\Scenario_07.zip" ^
  "data\Task 1\Scenario_08.zip" ^
  "data\Task 1\Scenario_09.zip" ^
  "data\Task 1\Scenario_10.zip" ^
  --prefix raw/Task1/ >> logs\task1_upload.log 2>&1

if errorlevel 1 (
  echo ----- pasada %ATTEMPT% fallo, reintento en 30s ----- >> logs\task1_upload.log
  REM timeout necesita una consola; ping al loopback funciona en procesos ocultos.
  ping -n 31 127.0.0.1 > nul
  goto retry
)

echo ===== TASK 1 COMPLETO - %DATE% %TIME% ===== >> logs\task1_upload.log
