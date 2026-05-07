@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo 首次运行：创建虚拟环境...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo 安装/更新依赖...
pip install -q -r requirements.txt

echo.
echo FaceRecall 启动中 → http://localhost:8787
echo 按 Ctrl+C 停止服务
echo.
python server.py
