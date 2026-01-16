#!/bin/bash
# 设置所有脚本的执行权限

echo "设置脚本执行权限..."

chmod +x install_dependencies.sh
chmod +x run_gui.sh
chmod +x build_linux.sh

echo "✓ 权限设置完成！"
echo ""
echo "现在可以运行："
echo "  ./install_dependencies.sh  # 安装依赖"
echo "  ./run_gui.sh               # 运行GUI"
echo "  ./build_linux.sh           # 打包应用"
