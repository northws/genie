@echo off
chcp 65001 > nul

REM 快速启动Genie GUI应用（开发模式）

echo 启动Genie GUI应用...
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python！
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo [警告] PyQt6未安装，正在安装...
    pip install PyQt6
    if errorlevel 1 (
        echo [错误] 安装PyQt6失败
        pause
        exit /b 1
    )
)

echo [2/3] 启动Genie GUI应用...
python genie_gui.py

echo [3/3] 应用已关闭
pause