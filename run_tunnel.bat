@echo off
title Morph ts - Cloudflare Tunnel for TripoSR
echo ============================================================
echo   Morph ts - Cloudflare HTTPS Tunnel for Local TripoSR GPU
echo ============================================================
echo.
cd /d "%~dp0"
"%~dp0cloudflared.exe" tunnel --url http://127.0.0.1:8001
pause
