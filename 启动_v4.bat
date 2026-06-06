@echo off
echo.
echo ================================================
echo    auto_learn v4.0 is starting, please wait...
echo    (first-time OCR model loading takes ~30s)
echo ================================================
echo.
py -3.11 "%~dp0auto_learn_v4.py"
pause
