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

python run_server.py --no-reload
pause
