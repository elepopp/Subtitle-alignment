@echo off
chcp 65001 >nul
rem 一次性安装：虚拟环境 + 全部依赖 + 下载全部模型到 models\
cd /d "%~dp0"
set UV_CACHE_DIR=%~dp0.uv-cache
where uv >nul 2>nul || (echo 需要 uv: https://docs.astral.sh/uv/ & pause & exit /b 1)
if not exist .venv\Scripts\python.exe uv venv .venv --python 3.11 || goto :err
uv pip install --python .venv\Scripts\python.exe torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || goto :err
uv pip install --python .venv\Scripts\python.exe -r webui\requirements.txt torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match || goto :err
uv pip install --python .venv\Scripts\python.exe --no-deps demucs==4.0.1 || goto :err
uv pip install --python .venv\Scripts\python.exe -e . --no-deps || goto :err
if not exist tools\ollama\ollama.exe (
  mkdir tools\ollama
  curl -L -o tools\ollama.zip https://github.com/ollama/ollama/releases/download/v0.35.1/ollama-windows-amd64.zip || goto :err
  tar -xf tools\ollama.zip -C tools\ollama && del tools\ollama.zip
)
.venv\Scripts\python.exe webui\download_models.py || goto :err
echo 完成。运行 start.bat 打开测试台。
pause
exit /b 0
:err
echo 安装失败
pause
exit /b 1
