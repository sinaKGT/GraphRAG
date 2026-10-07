@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title GraphRAG

where docker >nul 2>&1 || (echo [ERROR] Docker not found. Install Docker Desktop first. & pause & exit /b 1)
docker info >nul 2>&1 || (echo [ERROR] Docker Desktop is not running. Start it and run start.bat again. & pause & exit /b 1)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo [SETUP] Created .env - set NEO4J_PASSWORD, the providers, and GEMINI_API_KEY if you use Gemini. Save, then run start.bat again.
  notepad ".env"
  exit /b 1
)

rem ---- read settings from .env ----
set "APP_PORT=18400"
set "NEO4J_HTTP_PORT=18474"
set "NEO4J_BOLT_PORT=18687"
set "LLM_UI_PORT=18481"
set "NEO4J_PASSWORD="
set "LLM_PROVIDER=gemini"
set "EMBED_PROVIDER=gemini"
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
  if /i "%%A"=="APP_PORT" for /f "tokens=1" %%V in ("%%B") do set "APP_PORT=%%V"
  if /i "%%A"=="NEO4J_HTTP_PORT" for /f "tokens=1" %%V in ("%%B") do set "NEO4J_HTTP_PORT=%%V"
  if /i "%%A"=="NEO4J_BOLT_PORT" for /f "tokens=1" %%V in ("%%B") do set "NEO4J_BOLT_PORT=%%V"
  if /i "%%A"=="LLM_UI_PORT" for /f "tokens=1" %%V in ("%%B") do set "LLM_UI_PORT=%%V"
  if /i "%%A"=="NEO4J_PASSWORD" set "NEO4J_PASSWORD=%%B"
  if /i "%%A"=="LLM_PROVIDER" set "LLM_PROVIDER=%%B"
  if /i "%%A"=="EMBED_PROVIDER" set "EMBED_PROVIDER=%%B"
)

if not defined NEO4J_PASSWORD set "NEO4J_PASSWORD=x"
if "%NEO4J_PASSWORD:~7,1%"=="" (
  echo [ERROR] NEO4J_PASSWORD in .env must be at least 8 characters - Neo4j refuses shorter ones.
  notepad ".env"
  exit /b 1
)
findstr /c:"change-me-please" ".env" >nul && echo [WARN] NEO4J_PASSWORD is still the default - consider changing it.

set "USES_GEMINI="
set "PROFILE="
if /i "%LLM_PROVIDER%"=="gemini" set "USES_GEMINI=1"
if /i "%EMBED_PROVIDER%"=="gemini" set "USES_GEMINI=1"
if /i "%LLM_PROVIDER%"=="local" set "PROFILE=--profile local"
if /i "%EMBED_PROVIDER%"=="local" set "PROFILE=--profile local"

if defined USES_GEMINI (
  findstr /c:"PUT_YOUR_KEY_HERE" ".env" >nul && (
    echo [ERROR] A provider is set to gemini but GEMINI_API_KEY in .env is still the placeholder.
    notepad ".env"
    exit /b 1
  )
)
echo Providers: LLM=%LLM_PROVIDER%  embeddings=%EMBED_PROVIDER%

rem ---- make sure our ports are free (skipped when GraphRAG is already running) ----
set "RUNNING="
for /f %%I in ('docker ps -q -f "name=graphrag-" 2^>nul') do set "RUNNING=1"
if defined RUNNING goto ports_ok
call :checkport APP_PORT %APP_PORT% || goto port_busy
call :checkport NEO4J_HTTP_PORT %NEO4J_HTTP_PORT% || goto port_busy
call :checkport NEO4J_BOLT_PORT %NEO4J_BOLT_PORT% || goto port_busy
if defined PROFILE (call :checkport LLM_UI_PORT %LLM_UI_PORT% || goto port_busy)
goto ports_ok
:port_busy
pause
exit /b 1
:ports_ok

rem ---- local models: download first (foreground, so you can see progress) ----
if defined PROFILE (
  if not exist "models" mkdir "models"
  echo [0/3] Preparing local models in the Docker volume ^(one-time: copy from .\models or download the model files^)...
  docker compose %PROFILE% run --rm models-init
  if errorlevel 1 (echo [ERROR] Model download failed - run start.bat again to resume. & pause & exit /b 1)
)

echo [1/3] Building and starting containers...
docker compose %PROFILE% up -d --build
if errorlevel 1 (echo [ERROR] docker compose failed - see output above. & pause & exit /b 1)

echo [2/3] Waiting for backend on http://localhost:%APP_PORT% ...
set /a tries=0
:wait
curl -s -o nul http://localhost:%APP_PORT%/api/health && goto ready
set /a tries+=1
if %tries% geq 90 (
  echo [ERROR] Backend did not come up in time. Recent logs:
  docker compose %PROFILE% logs --tail 60
  pause
  exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait

:ready
echo [3/3] Ready.
start "" "http://localhost:%APP_PORT%"
echo.
echo   App:            http://localhost:%APP_PORT%
echo   Neo4j Browser:  http://localhost:%NEO4J_HTTP_PORT%   (user: neo4j, password from .env)
if defined PROFILE echo   llama.cpp UI:   http://localhost:%LLM_UI_PORT%   (chat with the local model directly)
echo   Stop:           stop.bat
echo.
echo Streaming logs - Ctrl+C stops watching, containers keep running.
if defined PROFILE (
  docker compose %PROFILE% logs -f backend llm embed
) else (
  docker compose logs -f backend
)
exit /b 0

rem ---- :checkport NAME PORT -> errorlevel 1 if another program is listening on PORT ----
:checkport
set "PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /c:":%~2 " ^| findstr LISTENING') do if not defined PID set "PID=%%P"
if not defined PID exit /b 0
set "PNAME=unknown"
for /f "tokens=1 delims=," %%N in ('tasklist /fi "PID eq %PID%" /fo csv /nh 2^>nul') do set "PNAME=%%~N"
echo.
echo [ERROR] Port %~2 is already in use by another program: %PNAME% ^(PID %PID%^).
echo         GraphRAG will not share a port. Pick a free one for %~1 in .env, then run start.bat again.
exit /b 1
