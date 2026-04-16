@echo off

REM 1. Activate Environment
call conda activate sniper_env

REM 2. STAY OPEN
echo.
echo ==========================================
echo SNIPER ENVIRONMENT ACTIVE
echo ==========================================
echo.

REM 3. Run Application

python daily_workflow.py
cmd /k