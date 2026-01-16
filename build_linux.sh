#!/bin/bash
# Genie GUI Linux打包脚本
# 将应用打包为独立可执行文件

echo "========================================"
echo "  Genie GUI - Linux打包工具"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3！"
    exit 1
fi

echo "当前Python版本："
python3 --version
echo ""

echo "[1/4] 检查依赖..."
python3 -c "import PyInstaller" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[警告] PyInstaller未安装，正在安装..."
    python3 -m pip install pyinstaller
    if [ $? -ne 0 ]; then
        echo "[错误] 安装PyInstaller失败"
        exit 1
    fi
fi

echo "[2/4] 清理旧构建..."
rm -rf build dist *.spec

echo "[3/4] 开始打包..."
python3 build_linux_pyinstaller.py
if [ $? -ne 0 ]; then
    echo "[错误] 打包失败"
    exit 1
fi

echo "[4/4] 完成！"
echo ""
echo "========================================"
echo "打包成功！"
echo "========================================"
echo ""
echo "输出位置: dist/Genie/"
echo "主程序: dist/Genie/Genie"
echo ""
echo "运行方式:"
echo "  cd dist/Genie"
echo "  ./Genie"
echo ""
echo "分发方式:"
echo "  tar -czf Genie-Linux-x64.tar.gz -C dist Genie"
echo ""
