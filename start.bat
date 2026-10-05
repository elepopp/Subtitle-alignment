@echo off
chcp 65001 >nul
rem subalign 功能测试台：启动 Web 页面 (http://127.0.0.1:7860)，同时启动项目内的 Ollama 本地大模型
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 未找到 .venv，请先运行 setup.bat
  pause
  exit /b 1
)
start "" http://127.0.0.1:7860
.venv\Scripts\python.exe -m webui.server %*
pause
