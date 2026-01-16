#!/bin/bash
# Genie GUI - 依赖安装脚本 (Linux/Mac)

echo "========================================"
echo "  Genie GUI - 安装依赖工具"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3！"
    echo ""
    echo "请先安装Python 3.8+："
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  Fedora/CentOS: sudo dnf install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

echo "当前Python版本："
python3 --version
echo ""

echo "[1/3] 升级pip..."
python3 -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "[警告] pip升级失败，继续安装..."
fi
echo ""

echo "[2/3] 安装GUI依赖..."
python3 -m pip install -r requirements_gui.txt
if [ $? -ne 0 ]; then
    echo "[错误] 依赖安装失败！"
    echo ""
    echo "请检查网络连接或尝试手动安装："
    echo "pip3 install PyQt6 numpy matplotlib pandas scipy scikit-learn"
    exit 1
fi
echo ""

echo "[3/3] 验证安装..."
python3 -c "import PyQt6; print('PyQt6: OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[错误] PyQt6安装验证失败"
    exit 1
fi

python3 -c "import torch; print('PyTorch: OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[警告] PyTorch未安装，某些功能可能不可用"
    echo ""
    echo "如需使用训练和采样功能，请安装PyTorch："
    echo "pip3 install torch torchvision torchaudio"
    echo ""
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "现在可以运行："
echo "  1. ./run_gui.sh 启动GUI应用"
echo "  2. ./build_linux.sh 打包为AppImage"
echo ""
