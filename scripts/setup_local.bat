@echo off
REM 一键起全本地环境：ollama 服务 + 拉模型 + 建索引 + 起应用
REM 用法：scripts\setup_local.bat
REM
REM 为什么需要这个脚本：ollama 默认 5 分钟不调用就卸载模型，
REM 下次请求要重新加载 2.5GB 权重，延迟从 0.4 秒跳到 17 秒。
REM 所以这里把服务拉起来并保持 keep_alive。

setlocal
cd /d "%~dp0.."

echo [1/4] 检查 ollama...
where ollama >nul 2>nul
if errorlevel 1 (
  echo     未找到 ollama。安装：scoop install ollama
  exit /b 1
)

REM 起 ollama 服务（如果已经在跑，这一步会报错但可以忽略）
echo [2/4] 启动 ollama 服务...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "127.0.0.1:11434" ^| findstr LISTENING') do (
  echo     服务已在运行 (PID %%p)，跳过
  goto :models
)
start "ollama" /min ollama serve
timeout /t 5 /nobreak >nul

:models
echo [3/4] 拉取模型（已存在则秒过）...
ollama pull bge-m3
ollama pull qwen2.5:7b

echo [4/4] 建索引...
call .venv\Scripts\python.exe scripts\seed_ops_db.py
call .venv\Scripts\python.exe scripts\build_index.py --rebuild

echo.
echo 完成。启动应用：uvicorn app.main:app --port 8000
echo 然后打开 http://127.0.0.1:8000/
endlocal
