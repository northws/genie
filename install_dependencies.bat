@echo off
chcp 65001 > nul

echo ========================================
echo   Genie GUI - 安装依赖工具
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python！
    echo.
    echo 请先安装Python 3.8+：
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo 当前Python版本：
python --version
echo.

echo [1/3] 升级pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [警告] pip升级失败，继续安装...
)
echo.

echo [2/3] 安装GUI依赖...
pip install -r requirements_gui.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败！
    echo.
    echo 请检查网络连接或尝试手动安装：
    echo pip install PyQt6 numpy matplotlib pandas scipy scikit-learn
    pause
    exit /b 1
)
echo.

echo [3/3] 验证安装...
python -c "import PyQt6; print('PyQt6: OK')" 2>nul
if errorlevel 1 (
    echo [错误] PyQt6安装验证失败
    pause
    exit /b 1
)

python -c "import torch; print('PyTorch: OK')" 2>nul
if errorlevel 1 (
    echo [警告] PyTorch未安装，某些功能可能不可用
    echo.
    echo 如需使用训练和采样功能，请安装PyTorch：
    echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo.
)

echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 现在可以运行：
echo   1. 双击 run_gui.bat 启动GUI应用
echo   2. 双击 build.bat 打包为EXE
echo.
pause
