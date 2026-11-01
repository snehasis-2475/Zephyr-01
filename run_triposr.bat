@echo off
title Sketch2Shape AI - Local TripoSR Engine
echo ============================================================
echo   Sketch2Shape AI - Local TripoSR Engine (RTX 4050 GPU)
echo ============================================================
echo.
cd /d "%~dp0"
set S2S_TRIPOSR_PORT=8001

:loop
echo [%time%] Starting TripoSR API server on port %S2S_TRIPOSR_PORT%...
"D:\IITBBSML\.venv\Scripts\python.exe" backend\triposr_server.py
echo.
echo [!] Server stopped. Restarting in 2 seconds...
timeout /t 2 /nobreak >nul
goto loop
