@echo off
chcp 65001 > nul

REM Genie GUI 自动打包脚本
REM 使用目录模式打包应用

echo ========================================
echo    Genie GUI 自动打包工具
echo ========================================
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python！请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

echo [1/4] 检查依赖...
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [警告] PyInstaller未安装，正在安装...
    pip install pyinstaller
    if errorlevel 1 (
        echo [错误] 安装PyInstaller失败
        pause
        exit /b 1
    )
)

echo [2/4] 清理旧构建...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "Genie.spec" del /q "Genie.spec"

echo [3/4] 开始打包...
python build_exe_optimized.py
if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo [4/4] 完成！
echo.
echo ========================================
echo 打包成功！
echo ========================================
echo.
echo 输出位置: dist\Genie\
echo 主程序: dist\Genie\Genie.exe
echo.
echo 您可以将 dist\Genie 文件夹压缩后分发
echo.
pause