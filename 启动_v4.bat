@echo off
chcp 65001 >nul
cd /d "%~dp0"
set OMP_NUM_THREADS=4
py -3.11 auto_learn_v4.py
pause
