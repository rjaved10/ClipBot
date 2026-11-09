@echo off
cd /d "%~dp0"
:loop
"C:\Users\booms\AppData\Local\Programs\Python\Python312\python.exe" bot.py
echo bot stopped, restarting in 30s
timeout /t 30 /nobreak >nul
goto loop
