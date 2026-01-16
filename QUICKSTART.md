# Genie GUI 快速开始指南

## 🚀 快速开始

### 第一步：安装依赖（必须！）

**⚠️ 重要：首次使用前必须先安装依赖！**

**选择A：使用安装脚本（最简单）**
```bash
双击运行: install_dependencies.bat
```
这会自动安装所有必需的依赖。

**选择B：手动安装**
```bash
# 安装GUI依赖
pip install -r requirements_gui.txt

# 或者只安装核心依赖
pip install PyQt6 numpy matplotlib pandas scipy scikit-learn torch pytorch-lightning
```

**验证安装：**
```bash
python -c "import PyQt6; print('安装成功！')"
```

### 第二步：运行应用

**安装完成后，启动GUI：**
```bash
双击运行: run_gui.bat
```

**或者命令行：**
```bash
python genie_gui.py
```

### 第三步：打包为EXE

**最简单方式：**
```bash
双击运行: build.bat
```

**或者手动执行：**
```bash
python build_exe_optimized.py
```

打包完成后，可执行文件位于：`dist/Genie/Genie.exe`

## 📦 打包输出

```
dist/Genie/
├── Genie.exe           # ← 双击这个启动应用
├── genie/              # 核心模块
├── evaluations/        # 评估工具
├── weights/            # 预训练模型（约2GB）
├── packages/           # 第三方工具
└── _internal/          # 运行时库
```

## 🎯 使用示例

### 示例1：使用预训练模型采样

1. 打开应用，切换到"采样"标签
2. 选择"使用预训练模型"
3. 从下拉菜单选择模型（如 `scope_l_128 (epoch=49999)`）
4. 设置参数：
   - 最小长度: 50
   - 最大长度: 128
   - 批次大小: 5
   - 批次数量: 2
5. 选择输出目录
6. 点击"开始采样"

### 示例2：评估生成的结构

1. 切换到"评估"标签
2. 选择输入目录（采样结果所在目录）
3. 选择输出目录
4. 勾选评估选项（全部勾选）
5. 点击"开始评估"

### 示例3：可视化结果

1. 切换到"绘图"标签
2. 选择绘图类型（如"全部图表"）
3. 选择输入目录（评估结果所在目录）
4. 选择输出目录
5. 点击"生成图表"
6. 双击列表中的图片查看

### 示例4：训练自定义模型

1. 切换到"训练"标签
2. 选择数据集或提供自定义数据集路径
3. 选择配置文件（.yml）
4. 设置训练参数
5. 点击"开始训练"

## ⚡ 性能提示

### GPU加速
在所有模块中设置GPU参数：
- 单GPU: `0`
- 多GPU: `0,1`
- CPU模式: 留空或删除

### 内存优化
- 训练：减小批次大小
- 采样：减小批次大小和批次数量
- 评估：一次处理较少文件

## 🔧 常见问题快速修复

### 问题：启动失败
```bash
# 重新安装依赖
pip install -r requirements_gui.txt --force-reinstall
```

### 问题：CUDA错误
```bash
# 安装CPU版本PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 问题：打包失败
```bash
# 清理并重试
rmdir /s /q build dist
del Genie.spec
python build_exe_optimized.py
```

### 问题：应用太大
- 删除不需要的预训练模型（weights目录）
- 使用目录模式而非单文件模式
- 修改 `build_exe_optimized.py` 排除更多模块

## 📊 预训练模型说明

| 模型名称 | 训练数据 | 最大长度 | 文件大小 | 推荐用途 |
|---------|---------|---------|---------|---------|
| scope_l_128 | SCOPE | 128 | ~800MB | 小蛋白质 |
| scope_l_256 | SCOPE | 256 | ~800MB | 中等蛋白质 |
| swissprot_l_256 | SwissProt | 256 | ~800MB | 通用蛋白质 |

## 📁 目录结构说明

```
genie/
├── genie_gui.py              # GUI主程序 ⭐
├── genie_backend.py          # 后端逻辑
├── build.bat                 # 一键打包 ⭐
├── run_gui.bat               # 一键运行 ⭐
├── build_exe_optimized.py    # 打包脚本（目录模式）
├── build_exe.py              # 打包脚本（单文件模式）
├── requirements_gui.txt      # GUI依赖
├── README_GUI.md             # 完整文档
├── QUICKSTART.md             # 本文件
├── genie/                    # 核心功能模块
├── evaluations/              # 评估和可视化工具
├── weights/                  # 预训练模型
└── packages/                 # 第三方工具
```

## 🎓 学习资源

- **完整文档**: 查看 `README_GUI.md`
- **原项目**: 查看 `README.md`
- **配置示例**: 查看 `weights/*/configuration`

## 💡 提示

1. **首次使用**：建议先在开发模式运行（`run_gui.bat`）测试功能
2. **打包前**：确保所有功能正常，weights目录完整
3. **分发时**：压缩整个 `dist/Genie/` 文件夹
4. **更新时**：重新运行 `build.bat` 即可

## ✅ 检查清单

打包前确认：
- [ ] 所有依赖已安装
- [ ] GUI可以正常运行
- [ ] weights目录包含所需模型
- [ ] 有足够磁盘空间（至少10GB）

分发前确认：
- [ ] 在干净系统上测试exe
- [ ] 检查所有功能可用
- [ ] 压缩文件完整
- [ ] 提供使用说明

## 🆘 获取帮助

如遇到问题：
1. 查看 `README_GUI.md` 的"常见问题"部分
2. 检查控制台输出的错误信息
3. 在GitHub仓库提issue

---

**祝使用愉快！** 🎉
