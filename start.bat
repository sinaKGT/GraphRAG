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
set "APP_PORT=8000"
set "NEO4J_PASSWORD="
set "LLM_PROVIDER=gemini"
set "EMBED_PROVIDER=gemini"
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
  if /i "%%A"=="APP_PORT" set "APP_PORT=%%B"
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
echo   Neo4j Browser:  http://localhost:7474   (user: neo4j, password from .env)
if defined PROFILE echo   llama.cpp UI:   http://localhost:8081   (chat with the local model directly)
echo   Stop:           stop.bat
echo.
echo Streaming logs - Ctrl+C stops watching, containers keep running.
if defined PROFILE (
  docker compose %PROFILE% logs -f backend llm embed
) else (
  docker compose logs -f backend
)
