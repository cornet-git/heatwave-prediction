@echo off
title HeatWatch API Server
echo.
echo  =========================================
echo   HeatWatch - Heatwave Risk Prediction API
echo  =========================================
echo   URL  : http://localhost:8000
echo   Docs : http://localhost:8000/docs
echo.
echo   Keep this window open while using the app.
echo   Press Ctrl+C to stop the server.
echo  =========================================
echo.

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3.10 run_server.py --no-reload
) else (
    python run_server.py --no-reload
)
pause
