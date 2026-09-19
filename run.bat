@echo off
setlocal

rem Runs the agentic-rl dev server. From the repo root: run.bat [port]
rem .env (if present) is loaded automatically by the app itself (python-dotenv,
rem see api/app.py) — this script doesn't need to parse it.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run.bat] No .venv found. Set one up first:
    echo   python -m venv .venv
    echo   .venv\Scripts\python -m pip install -e ".[dev]"
    echo   ^(add ".[dev,observability]" instead if you want Langfuse tracing^)
    exit /b 1
)

if not exist ".env" (
    echo [run.bat] No .env found ^(copy .env.example to .env to configure it^).
    echo [run.bat] Continuing with defaults: AGENTIC_RL_PLANNER=claude needs
    echo [run.bat] ANTHROPIC_API_KEY set some other way ^(openai/google work too,
    echo [run.bat] set AGENTIC_RL_PLANNER + the matching *_API_KEY^), or use
    echo [run.bat] AGENTIC_RL_PLANNER=mock to skip needing a key at all.
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

echo [run.bat] Starting agentic-rl on http://127.0.0.1:%PORT% ...
".venv\Scripts\python.exe" -m uvicorn agentic_rl.api.app:create_app --factory --host 127.0.0.1 --port %PORT%
