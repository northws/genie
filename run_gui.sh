#!/bin/bash
# 快速启动Genie GUI应用（开发模式）

echo "启动Genie GUI应用..."
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3！"
    exit 1
fi

# 检查PyQt6
python3 -c "import PyQt6" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[警告] PyQt6未安装，正在安装..."
    python3 -m pip install PyQt6
fi

# 运行GUI（优先使用增强版）
if [ -f "genie_gui_enhanced.py" ]; then
    python3 genie_gui_enhanced.py
elif [ -f "genie_gui.py" ]; then
    python3 genie_gui.py
else
    echo "[错误] 未找到GUI文件！"
    exit 1
fi
