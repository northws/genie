# Genie 项目优化与加速指南 (Optimization Guide)

在复现文献的过程中针对 Genie 蛋白质设计模型加入了少量的优化措施，其中涵盖数据 I/O、训练速度、显存管理及 FlashIPA方法的集成。

## 1. 核心优化概览

| 优化模块 | 优化手段 | 预期收益 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **数据加载** | 二进制 `.npy` 格式替代文本 CSV | **I/O 速度提升 10x+** | 所有场景 |
| **优化器** | 启用 Fused Adam | **训练速度提升 5-10%** | GPU 训练 |
| **卷积计算** | 启用 cuDNN Benchmark | **卷积层计算加速** | 固定输入尺寸 |
| **注意力机制** | 集成 FlashIPA (FlashAttention) | **显存降低，长序列加速** | 序列长度 > 512 |
| **Transform 模块** | 可配置梯度检查点 (Gradient Checkpointing) | **训练速度提升 20-30%** | 显存充足时 (N <= 256) |

---

## 2. 详细说明

### 2.1 数据 I/O 优化
*   原项目使用 `np.savetxt` 存储坐标，读取极慢。我们在此处将其改为 `np.save` 存储二进制文件。
*   **代码修改**: `scripts/generate_scope_coords.py` 和 `genie/utils/data_io.py`。

### 2.2 FlashIPA 的集成
*   引入 `flash_ipa` 库，利用 FlashAttention 加速 Invariant Point Attention (IPA) 计算。
*   标准 IPA 计算非常昂贵，因为它需要计算所有残基对之间的相互作用，计算复杂度和显存占用通常是 $O (N^{2})$（N 为序列长度）。FlashIPA使用了 Tiling（分块） 技术，避免了显存中存储巨大的 $N×N$ 注意力矩阵。并引入了 z_factor（低秩分解），将原本庞大的成对特征表示进行压缩，大幅减少显存占用。
*   **针对序列长度选取**由于在FlashIPA中序列长度小于512时GPU run time相比OrigIPA更长:
    *   在 `genie/model/structure_net.py` 中加入了对输入序列的长度判断。
    *   当序列长度 `max_n_res <= 512` 时，自动**关闭** FlashIPA 以避免 Kernel 启动开销（此时标准实现更快）。
    *   当序列长度 `max_n_res > 512` 时，自动**开启** FlashIPA 以节省显存并加速。
*   **使用**: 在 `config.yaml` 中设置 `useFlashIPA: True` 即可启用此智能逻辑。

### 2.3 Transform 模块加速 
*    `PairTransformNet` 中的三角形更新 ($O(N^3)$) 默认开启了梯度检查点 (`checkpoint`)，导致反向传播时重复计算。故引入 `useGradientCheckpointing` 配置。
*    现在更改为
    *   **默认 (False)**: 关闭检查点，**消除重计算，训练显著加速**（适用于 $N=128&256$）。
    *   **开启 (True)**: 节省显存，但速度较慢（适用于超长序列 N > 512 且显存不足时）。

### 2.4 训练环境优化
*   **Fused Adam**: 自动检测 GPU 环境并启用 `fused=True`，加速参数更新。
*   **cuDNN Benchmark**: 启用 `torch.backends.cudnn.benchmark = True`，自动寻找最优卷积算法。
*   **混合精度**: 保持 `precision='16-mixed'`，利用 Tensor Cores。
*   **显存碎片**: 设置 `PYTORCH_ALLOC_CONF=expandable_segments:True` 减少 OOM 风险。

---

## 3. 配置文件示例 

推荐的配置如下（针对 N=128 的标准训练）：

```config
name: final_optimized
numEpoches: 500
batchSize: 8
maximumNumResidues: 128  # 短序列，FlashIPA 会自动关闭   
dataDirectory: data
datasetNames: scope
templateType: v1
numPairTransformLayers: 8
includeTriangularAttention: True
logEverySteps: 50
checkpointEveryEpoches: 50
learningRate: 2e-4
useFlashIPA: True        # 保持开启，由代码自动判断是否介入
useGradientCheckpointing: False # 关闭以获得最大速度，开启能节省显存
numWorkers: 8            # 根据 CPU 核心数调整
```

## 4. 常见问题

*   **Q: 如何查看 FlashIPA 是否启动？**
    *   A: 如果 `maximumNumResidues` <= 512，我们在这里设置了一个 `Info` 提示并回退到标准实现，如果未显示info且在config文件中设置useFlashIPA: True，则FlashIPA默认开启。
