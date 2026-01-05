# Genie 项目优化与加速指南 (Optimization Guide)

本文档详细记录了针对 Genie 蛋白质设计模型的全方位优化措施，涵盖数据 I/O、训练速度、显存管理及 FlashIPA 集成。

## 1. 核心优化概览

| 优化模块 | 优化手段 | 预期收益 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **数据加载** | 二进制 `.npy` 格式替代文本 CSV | **I/O 速度提升 10x+** | 所有场景 |
| **优化器** | 启用 Fused Adam | **训练速度提升 5-10%** | GPU 训练 |
| **卷积计算** | 启用 cuDNN Benchmark | **卷积层计算加速** | 固定输入尺寸 |
| **注意力机制** | 集成 FlashIPA (FlashAttention) | **显存降低，长序列加速** | 序列长度 > 512 |
| **Transform 模块** | 可配置梯度检查点 (Gradient Checkpointing) | **训练速度提升 20-30%** | 显存充足时 (N <= 256) |

---

## 2. 详细优化说明

### 2.1 数据 I/O 优化 (Data I/O)
*   **原理**: 原项目使用 `np.savetxt` 存储坐标，读取极慢。我们改为 `np.save` 存储二进制文件。
*   **代码修改**: `scripts/generate_scope_coords.py` 和 `genie/utils/data_io.py`。
*   **操作建议**: 运行 `python scripts/generate_scope_coords.py` 重新生成数据以生效。

### 2.2 FlashIPA 集成 (FlashIPA Integration)
*   **原理**: 引入 `flash_ipa` 库，利用 FlashAttention 加速 Invariant Point Attention (IPA) 计算。
*   **智能切换**:
    *   在 `genie/model/structure_net.py` 中实现了智能逻辑。
    *   当序列长度 `max_n_res <= 512` 时，自动**关闭** FlashIPA 以避免 Kernel 启动开销（此时标准实现更快）。
    *   当序列长度 `max_n_res > 512` 时，自动**开启** FlashIPA 以节省显存并加速。
*   **配置**: 在 `config.yaml` 中设置 `useFlashIPA: True` 即可启用此智能逻辑。

### 2.3 Transform 模块加速 (PairTransformNet Speedup)
*   **原理**: `PairTransformNet` 中的三角形更新 ($O(N^3)$) 默认开启了梯度检查点 (`checkpoint`)，导致反向传播时重复计算。
*   **优化**: 引入 `useGradientCheckpointing` 配置。
    *   **默认 (False)**: 关闭检查点，**消除重计算，训练显著加速**（适用于 N=128/256）。
    *   **开启 (True)**: 节省显存，但速度较慢（适用于超长序列 N > 512 且显存不足时）。
*   **配置**: 在 `config.yaml` 中添加 `useGradientCheckpointing: False` (默认即为 False)。

### 2.4 训练环境优化 (Training Environment)
*   **Fused Adam**: 自动检测 GPU 环境并启用 `fused=True`，加速参数更新。
*   **cuDNN Benchmark**: 启用 `torch.backends.cudnn.benchmark = True`，自动寻找最优卷积算法。
*   **混合精度**: 保持 `precision='16-mixed'`，利用 Tensor Cores。
*   **显存碎片**: 设置 `PYTORCH_ALLOC_CONF=expandable_segments:True` 减少 OOM 风险。

---

## 3. 配置文件示例 (config.yaml)

推荐的配置如下（针对 N=128 的标准训练）：

```yaml
name: final_optimized
numEpoches: 500
batchSize: 8
maximumNumResidues: 128  # 短序列，FlashIPA 会自动休眠
dataDirectory: data
datasetNames: scope
templateType: v1
numPairTransformLayers: 8
includeTriangularAttention: True
logEverySteps: 50
checkpointEveryEpoches: 50
learningRate: 2e-4
useFlashIPA: True        # 保持开启，由代码自动判断是否介入
useGradientCheckpointing: False # 关键：关闭以获得最大速度
numWorkers: 8            # 根据 CPU 核心数调整
```

## 4. 常见问题

*   **Q: 为什么我看不到 FlashIPA 启动日志？**
    *   A: 如果您的 `maximumNumResidues` <= 512，系统会打印一条 `Info` 提示并回退到标准实现，这是为了保证速度。
*   **Q: 显存不够了怎么办？**
    *   A: 将 `useGradientCheckpointing` 设为 `True`，或者减小 `batchSize`。
