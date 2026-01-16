# Genie GUI 跨平台支持

## 🌍 支持的平台

Genie GUI现在支持以下平台：

| 平台 | 状态 | 打包格式 |
|------|------|---------|
| Windows 10/11 | ✅ 完全支持 | EXE (PyInstaller) |
| Linux (x64) | ✅ 完全支持 | 可执行文件 + tar.gz |
| macOS | ✅ 完全支持 | App Bundle |

## 📦 Windows 安装和使用

### 安装依赖
```bash
# 双击运行
install_dependencies.bat

# 或命令行
pip install -r requirements_gui.txt
```

### 运行应用
```bash
# 双击运行
run_gui.bat

# 或命令行
python genie_gui_enhanced.py
```

### 打包为EXE
```bash
# 双击运行
build.bat

# 或命令行
python build_exe_optimized.py
```

### 分发
```
打包后：
1. 压缩 dist\Genie\ 文件夹
2. 分发 zip 文件
3. 用户解压后运行 Genie.exe
```

## 🐧 Linux 安装和使用

### 安装依赖
```bash
# 添加执行权限
chmod +x install_dependencies.sh

# 运行安装脚本
./install_dependencies.sh

# 或手动安装
pip3 install -r requirements_gui.txt
```

### 运行应用
```bash
# 添加执行权限
chmod +x run_gui.sh

# 运行应用
./run_gui.sh

# 或直接运行
python3 genie_gui_enhanced.py
```

### 打包为可执行文件
```bash
# 添加执行权限
chmod +x build_linux.sh

# 运行打包脚本
./build_linux.sh

# 或手动打包
python3 build_linux_pyinstaller.py
```

### 分发
```bash
# 打包为tar.gz
tar -czf Genie-Linux-x64.tar.gz -C dist Genie

# 分发tar.gz文件
# 用户解压后：
tar -xzf Genie-Linux-x64.tar.gz
cd Genie
./Genie
```

## 🍎 macOS 安装和使用

### 安装依赖
```bash
# 添加执行权限
chmod +x install_dependencies.sh

# 运行安装脚本
./install_dependencies.sh

# 或使用Homebrew
brew install python3
pip3 install -r requirements_gui.txt
```

### 运行应用
```bash
# 添加执行权限
chmod +x run_gui.sh

# 运行应用
./run_gui.sh

# 或直接运行
python3 genie_gui_enhanced.py
```

### 打包为App
```bash
# 使用build_linux.sh（同样适用于macOS）
chmod +x build_linux.sh
./build_linux.sh
```

### 分发
```bash
# 创建DMG（需要额外工具）
# 或简单打包为tar.gz
tar -czf Genie-macOS.tar.gz -C dist Genie
```

## 🔧 平台特定问题

### Windows

**问题：缺少CUDA**
```bash
# 安装CPU版本PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**问题：exe文件被杀毒软件阻止**
- 添加到信任列表
- 或使用目录模式而非单文件模式

### Linux

**问题：缺少系统库**
```bash
# Ubuntu/Debian
sudo apt install python3-dev libgl1-mesa-glx

# Fedora/CentOS
sudo dnf install python3-devel mesa-libGL

# Arch Linux
sudo pacman -S python mesa
```

**问题：Qt平台插件错误**
```bash
# 安装Qt依赖
sudo apt install libxcb-xinerama0 libxcb-cursor0

# 或设置环境变量
export QT_QPA_PLATFORM=xcb
```

### macOS

**问题：应用无法打开**
```bash
# 允许运行未签名应用
xattr -cr dist/Genie/Genie
```

**问题：缺少命令行工具**
```bash
# 安装Xcode命令行工具
xcode-select --install
```

## 📊 功能对比

| 功能 | Windows | Linux | macOS |
|------|---------|-------|-------|
| 基础GUI | ✅ | ✅ | ✅ |
| 训练模块 | ✅ | ✅ | ✅ |
| 采样模块 | ✅ | ✅ | ✅ |
| 评估模块 | ✅ | ✅ | ✅ |
| 绘图模块 | ✅ | ✅ | ✅ |
| 文件拖放 | ✅ | ✅ | ✅ |
| 配置编辑 | ✅ | ✅ | ✅ |
| GPU加速 | ✅ | ✅ | ✅ |
| 任务队列 | ✅ | ✅ | ✅ |

## 🚀 性能优化

### 所有平台

```python
# GPU设置
GPU设备: 0          # 单GPU
GPU设备: 0,1        # 多GPU
GPU设备: (留空)    # CPU模式
```

### Windows特定
- 使用CUDA 11.8+获得最佳性能
- 启用硬件加速

### Linux特定
- 安装最新NVIDIA驱动
- 使用nvidia-docker（容器环境）

### macOS特定
- M1/M2芯片使用MPS后端
- Intel芯片使用CPU模式

## 📝 环境变量

### Windows
```batch
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set OMP_NUM_THREADS=8
```

### Linux/macOS
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export QT_QPA_PLATFORM=xcb  # Linux专用
```

## 🐳 Docker支持

### 构建镜像
```bash
# 创建Dockerfile（参见docker/Dockerfile）
docker build -t genie-gui .
```

### 运行容器
```bash
# Linux with GPU
docker run --gpus all -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    genie-gui

# macOS (需要XQuartz)
xhost + localhost
docker run -e DISPLAY=host.docker.internal:0 \
    genie-gui
```

## 🔄 升级和更新

### 所有平台
```bash
# 更新依赖
pip install -r requirements_gui.txt --upgrade

# 更新应用
git pull origin main

# 重新打包
# Windows: build.bat
# Linux/macOS: ./build_linux.sh
```

## 📞 获取帮助

### 常见问题
- **Windows**: 查看 `README_GUI.md`
- **Linux**: 查看 `/usr/share/doc/genie-gui/`
- **macOS**: 查看 Application Support

### 社区支持
- GitHub Issues
- 讨论区
- 邮件列表

## 📜 许可证

遵循原项目许可证（见LICENSE.md）

## 🙏 贡献

欢迎为跨平台支持做出贡献：
- 报告平台特定问题
- 提交兼容性修复
- 改进打包脚本
- 完善文档

---

**跨平台支持版本**: v1.1.0
**更新日期**: 2026-01-17
