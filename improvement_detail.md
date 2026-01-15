# Genie Optimization Details

## 概述

本文档详细记录了 genie（下面称为**genie-v1**以区分） 相对于原版 genie 的所有优化改进点，包括训练效率、内存优化、代码结构优化等方面。

---

## 1. 训练优化 (train.py)

### 1.1 内存优化

**原版 genie/train.py:**
```python
# 无内存优化配置
```

**genie-v1/train.py:**

```python
# [Optimization] Reduce memory fragmentation
# Setting this before importing torch to ensure it takes effect
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

...

# [Added] Monitor CUDA memory allocation globally
try:
    torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory for system
except RuntimeError:
    pass  # Not applicable on CPU-only
```

### 1.2 Tensor Core 和 TF32 支持

**genie-v1/train.py:**
```python
# [Optimization] Enable TF32 on Ampere+ GPUs (A100, RTX3090, etc.)
# 'medium' or 'high' enables Tensor Cores for float32 matrix multiplications
torch.set_float32_matmul_precision('medium')

...

# [Optimization] Tensor Core Support
# '16-mixed' uses FP16 for matmul (Tensor Cores) and FP32 for stability.
# Use 'bf16-mixed' if you are on NVIDIA GPUs starting from A100 for better stability.
precision='bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed',
```

### 1.3 cuDNN Benchmark

**genie-v1/train.py:**
```python
# [Optimization] Enable cuDNN benchmark for fixed input sizes
# This finds the best convolution algorithms for the hardware
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

### 1.4 梯度裁剪优化

**原版 genie/train.py:**
```python
gradient_clip_val=1.0,
```

**genie-v1/train.py:**
```python
# [Stability] Gradient Clipping disabled to allow Fused AdamW
gradient_clip_val=None,
```

### 1.5 新增参数

**genie-v1/train.py:**
```python
# 新增 num_workers 参数
dm = SCOPeDataModule(**config.io, batch_size=config.training['batch_size'], num_workers=config.training['num_workers'])

# 新增 OOM 监控回调
callbacks=[checkpoint_callback, OOMMonitorCallback()]

# 新增 resume 参数支持
parser.add_argument('-r', '--resume', type=str, help='Path for checkpoint file to resume from')
```

---

## 2. 配置优化 (config.py)

### 2.1 新增配置项

**genie-v1/config.py:**
```python
self.model = {
    ...
    # [Optimization] 新增配置项
    'use_flash_ipa': config.get('useFlashIPA', True),
    'max_n_res': self.io['max_n_res'],
    'use_grad_checkpoint': config.get('useGradientCheckpointing', False)
}

self.training = {
    ...
    # 新增 num_workers
    'num_workers': int(config.get('numWorkers', 4)),
}
```

---

## 3. 模型优化 - IPA 模块

### 3.1 Gradient Checkpointing

**原版 genie/model/modules/invariant_point_attention.py:**

```python
class InvariantPointAttention(nn.Module):
    def __init__(self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, inf=1e5, eps=1e8):
```

**genie-v1/model/modules/invariant_point_attention.py:**
```python
class InvariantPointAttention(nn.Module):
    def __init__(self, ..., use_checkpointing=True):
        ...
        self.use_checkpointing = use_checkpointing

    def _run_ipa(self, s, z, t_trans, t_rots, mask):
        # 封装 IPA 核心计算逻辑
        ...

    def forward(self, s, z, t, mask):
        # Optimization: Apply checkpointing
        if self.training and s.requires_grad and self.use_checkpointing:
            return checkpoint(self._run_ipa, s, z, t.trans, t.rots, mask, use_reentrant=False)
        else:
            return self._run_ipa(s, z, t.trans, t.rots, mask)
```

---

## 4. 新增工具 - OOM Monitor Callback

**genie-v1/utils/oom_callback.py:**
```python
class OOMMonitorCallback(pl.Callback):
    """
    Callback that attempts to catch and log OutOfMemory errors,
    and prints memory stats before crashing.
    """
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        try:
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated()
                mem_res = torch.cuda.memory_reserved()
                max_mem = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            pass

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, RuntimeError) and "out of memory" in str(exception).lower():
            print("\n" + "="*80)
            print("CRITICAL ERROR: CUDA OUT OF MEMORY (OOM) DETECTED")
            ...
```

---

## 5. Evaluation 优化 (pipeline)

### 5.1 GPU 配置和兼容性

**evaluations-v1/pipeline/evaluate.py:**
```python
import types
import sys

# HACK: Patch torch._six for deepspeed compatibility
if not hasattr(torch, '_six'):
    torch._six = types.ModuleType('torch._six')
    torch._six.inf = torch.inf
    sys.modules['torch._six'] = torch._six

# Add current directory to path so we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Initialize models with显式设备管理
    fold_model = ESMFold()
    inverse_fold_model = ProteinMPNN(device='cuda:0')
```

**evaluations/pipeline/evaluate.py:**
```python
# 无 torch._six 补丁
inverse_fold_model = ProteinMPNN()
fold_model = ESMFold()
```

### 5.2 错误处理

**evaluations-v1/pipeline/evaluate.py:**
```python
try:
    main(args)
except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print('\n' + '='*60)
        print('CRITICAL ERROR: CUDA Out of Memory (OOM) during evaluation.')
        ...
        sys.exit(1)
    else:
        raise e
```

**evaluations/pipeline/evaluate.py:**
```python
main(args)  
```

### 5.3 新增命令行参数

**evaluations-v1/pipeline/evaluate.py:**

```python
parser.add_argument('-g', '--gpus', type=str, default=None, help='GPU devices to use (e.g., "0,1")')
parser.add_argument('-c', '--config', type=str, help='Config file (ignored but accepted for compatibility)')
```

---

## 6. Novelty Evaluation  (为了评估新颖性自己写的，包含一个CPU版一个GPU版)

### 6.1 混合 GPU 筛选方法

**evaluations-v1/Novelty_Evaluation_GPU.py** 包含以下内容：

```python
# --- Configuration ---
K_NEIGHBORS = 30 
BATCH_SIZE = 25 
TOP_K_SCREEN = 1000  # Increased from TOP_K_SCREEN for better recall
SIM_CHUNK_SIZE = 5000  # Chunk size for similarity computation to avoid OOM

# Early stopping optimization
EARLY_STOP_TM = 0.5  # Stop early if TM-score exceeds this threshold
ENABLE_EARLY_STOP = True  # Enable early stopping for faster novelty detection

# Length filtering optimization
LENGTH_TOLERANCE = 0.3  # Only compare structures with length within ±30%
ENABLE_LENGTH_FILTER = False  # Enable length-based pre-filtering
```

### 6.2 GPU 加速嵌入计算

**genie-v1/Novelty_Evaluation_GPU.py:**
```python
def compute_embeddings(model, pdb_files):
    all_embeddings = []
    valid_files_out = []
    
    chunks = [pdb_files[i:i + BATCH_SIZE] for i in range(0, len(pdb_files), BATCH_SIZE)]
    
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Embedding Batch"):
            # 批量处理 PDB 文件
            # 使用 GPU 加速 ProteinMPNN 嵌入计算
            ...
    
    return torch.cat(all_embeddings, dim=0), valid_files_out
```

### 6.3 相似度矩阵分块计算

**genie-v1/Novelty_Evaluation_GPU.py:**
```python
# 3. Compute Similarity Matrix (chunked to avoid OOM)
print("Computing Similarity Matrix (chunked)...")

# Chunked computation for large matrices
all_top_idxs = []
all_top_vals = []

for i in range(0, num_designs, SIM_CHUNK_SIZE):
    end_i = min(i + SIM_CHUNK_SIZE, num_designs)
    design_chunk = design_embs[i:end_i]
    
    # Compute similarity for this chunk
    sim_chunk = torch.matmul(design_chunk, ref_embs.T)
    
    # Get top-k for this chunk
    top_vals_chunk, top_idxs_chunk = torch.topk(sim_chunk, k=actual_top_k, dim=1)
    all_top_idxs.append(top_idxs_chunk.cpu())
    all_top_vals.append(top_vals_chunk.cpu())
    
    # Free memory
    del sim_chunk
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

### 6.4 多进程 TMalign 验证

**genie-v1/Novelty_Evaluation_GPU.py:**
```python
# 4. Top-K Selection and Parallel TMalign
N_WORKERS = 20
print(f"Running TMalign verification with {N_WORKERS} workers...")

worker = functools.partial(process_design, design_paths=design_names, 
                           ref_paths_all=ref_names, candidate_indices_list=top_idxs, 
                           length_tolerance=LENGTH_TOLERANCE, 
                           enable_length_filter=ENABLE_LENGTH_FILTER)

results = []
with Pool(N_WORKERS) as p:
    for res_str in tqdm(p.imap_unordered(worker, indices), total=num_designs):
        results.append(res_str)
```

### 6.5 早停（>0.5便可认为不新颖）优化

**genie-v1/Novelty_Evaluation_GPU.py:**
```python
def process_design(...):
    for ref_p in candidate_refs:
        ...
        tm = float(match.group(1))
        if tm > max_tm:
            max_tm = tm
            best_ref = ref_p
            
            # Early stopping: if TM > threshold, structure is NOT novel
            if ENABLE_EARLY_STOP and tm > EARLY_STOP_TM:
                break  # 提前终止搜索
```

### 6.6 新增评估脚本

| 脚本 | 描述 |
|------|------|
| `Novelty_Evaluation_GPU.py` | GPU 加速的新颖性评估 |
| `Novelty_Evaluation_CPU.py` | CPU 版本的新颖性评估 |
| `visualize_protein.py` | 蛋白质结构可视化 |
| `visualize_trajectory.py` | 训练轨迹可视化 |
| `visualize.py` | 综合可视化工具 |
| `plot.py` | 绘图工具 |

---

## 7. Utils 工具优化

### 7.1 工具函数对比

**evaluations-v1/pipeline/utils.py** 与 **evaluations/pipeline/utils.py** 基本相同，但 evaluations-v1 版本新增了更多辅助函数：

- `hcluster()` - 层次聚类
- `save_as_pdb()` - PDB 文件保存
- `parse_tm_file()` - TM-score 文件解析
- `parse_pdb_file()` - PDB 文件解析
- `parse_pae_file()` - PAE 文件解析
- `distance()`, `angle()`, `dihedral()` - 几何计算
- `assign_secondary_structures()` - 二级结构分配
- `assign_left_handed_helices()` - 左手螺旋检测

---

## 8. 性能对比总结

### 8.1 训练效率提升

| 优化项 | 预期效果 |
|--------|----------|
| TF32/BF16 混合精度 | 2-3x 训练速度提升 |
| cuDNN Benchmark | 固定输入尺寸下 10-30% 加速 |
| expandable_segments | 减少内存碎片，提高显存利用率 |
| Fused AdamW (无梯度裁剪) | 减少内存占用和计算开销 |
| Gradient Checkpointing | 显著降低显存，允许更大 batch size |

### 8.2 稳定性提升

| 优化项 | 效果 |
|--------|------|
| OOMMonitorCallback | 快速定位 OOM 问题 |
| torch._six 补丁 | 解决 DeepSpeed 兼容性问题 |
| BF16 支持 | A100 等之后的显卡上更稳定 |
| 显式 GPU 设备管理 | 避免设备冲突 |

---

## 9. 使用建议

### 9.1 推荐配置

```bash
# A100/H100 或更高专业显卡
    --precision bf16-mixed

# RTX 3090/4090/5090 等新架构消费级显卡
    --precision 16-mixed
```

### 9.2 配置文件示例

```
# config.txt
useFlashIPA True
useGradientCheckpointing True
numWorkers 8
batchSize 64
```

---

## 10. 几何工具向量化优化 (geo_utils.py)

### 10.1 TorchScript JIT 编译

**genie-v1/utils/geo_utils.py:**
```python
import torch
import numpy as np

@torch.jit.script
def distance(p, eps: float = 1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5

@torch.jit.script
def dihedral(p, eps: float = 1e-10):
    # 使用 TorchScript 编译，加速执行
    ...
    return torch.stack([cos_enc, sin_enc], dim=-1)

@torch.jit.script
def compute_frenet_frames(x, mask, eps: float = 1e-10):
    # 向量化计算，完全消除 Python 循环
    ...
```

**genie/utils/geo_utils.py:**
```python
import torch
import numpy as np

def distance(p, eps=1e-10):
    # 无 JIT 编译
    ...

def dihedral(p, eps=1e-10):
    # 无 JIT 编译
    ...

def compute_frenet_frames(x, mask, eps=1e-10):
    # 包含 Python 循环 for i in range(mask.shape[0])
    ...
```

### 10.2 Frenet Frames 向量化计算

**原版 genie/utils/geo_utils.py - 使用 Python 循环:**
```python
def compute_frenet_frames(x, mask, eps=1e-10):
    # x: [b, n_res, 3]
    
    t = x[:, 1:] - x[:, :-1]
    ...
    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # [Optimization TODO] 使用 Python 循环，效率低
    rots = []
    for i in range(mask.shape[0]):  # 逐个 batch 处理
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1:length-1] = tbn[i, :length-2]
        rots_[0] = rots_[1]
        rots_[length-1] = rots_[length-2]
        rots.append(rots_)
    rots = torch.stack(rots, dim=0).to(x.device)

    return rots
```

**genie-v1/utils/geo_utils.py - 完全向量化:**
```python
def compute_frenet_frames(x, mask, eps: float = 1e-10):
    """
    Vectorized computation of Frenet-Serret frames.
    x: [b, n_res, 3]
    mask: [b, n_res]
    """
    # 向量化计算 tangent, binormal, normal
    t = x[:, 1:] - x[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    b = torch.cross(t[:, :-1], t[:, 1:], dim=-1)
    b_norm = torch.sqrt(eps + torch.sum(b ** 2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    n = torch.cross(b, t[:, 1:], dim=-1)
    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    B, N, _ = x.shape
    device = x.device

    # [Optimization] 移除 Python 循环，使用纯张量操作
    rots = torch.eye(3, device=device, dtype=x.dtype).view(1, 1, 3, 3).repeat(B, N, 1, 1)

    # 1. 填充中间帧
    rots[:, 1:-1] = tbn

    # 2. 处理 N 端
    rots[:, 0] = rots[:, 1]

    # 3. 处理 C 端
    lengths = mask.sum(dim=1).long()  # [B]
    batch_indices = torch.arange(B, device=device)
    src_indices = (lengths - 2).clamp(min=0)
    tgt_indices = (lengths - 1).clamp(min=0)
    c_term_frames = rots[batch_indices, src_indices]
    rots[batch_indices, tgt_indices] = c_term_frames

    # 4. 清理填充区域
    mask_expanded = mask.view(B, N, 1, 1)
    identity = torch.eye(3, device=device, dtype=x.dtype).view(1, 1, 3, 3)
    rots = rots * mask_expanded + identity * (1 - mask_expanded)

    return rots
```

### 10.3 向量化优化效果

| 特性 | Python 循环版本 | 向量化版本 |
|------|-----------------|------------|
| 代码行数 | ~25 行 | ~40 行 |
| Python 循环 | `for i in range(batch)` | 无 |
| GPU 利用率 | 低 | 高 |
| 执行速度 | 慢 | 快 (~10x) |
| 内存效率 | 低 | 高 |
| TorchScript | 无 | 支持 |

### 10.4 关键优化点

1. **移除 Python 循环**: 逐 batch 循环改为纯张量操作
2. **TorchScript JIT**: 使用 `@torch.jit.script` 装饰器编译函数
3. **批量索引操作**: 使用 `torch.arange` 和 `gather/scatter` 代替循环
4. **掩码处理**: 使用张量操作处理填充区域

---

## 11. 数据加载优化 (data_io.py)

### 11.1 文件路径缓存机制

**genie-v1/utils/data_io.py - 使用缓存:**
```python
def load_filepaths(datadir, dataset_names, max_n_res=None, min_n_res=None, classes=None, n_data=None):
    # [Optimization] 检查缓存的文件列表，避免在网络驱动或大型数据集上缓慢的 glob
    cache_key = f"{'_'.join(dataset_names)}_min{min_n_res}_max{max_n_res}_cls{classes is None}.pkl"
    cache_path = os.path.join(datadir, 'cache', cache_key)

    if os.path.exists(cache_path):
        print(f"Loading filepaths from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            output_filepaths = pickle.load(f)
            if n_data is not None and n_data < len(output_filepaths):
                return output_filepaths[:n_data]
            return output_filepaths

    # ... 执行实际的 glob 和过滤 ...

    # 保存缓存
    os.makedirs(os.path.join(datadir, 'cache'), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(output_filepaths, f)

    return output_filepaths
```

**genie/utils/data_io.py - 无缓存:**
```python
def load_filepaths(datadir, dataset_names, max_n_res=None, min_n_res=None, classes=None, n_data=None):
    # 每次启动训练时重复扫描文件
    output_filepaths = []
    ...
    # 无缓存机制，每次都执行完整的 glob 和过滤
    return output_filepaths
```

### 11.2 坐标文件加载优化

**genie-v1/utils/data_io.py - 智能加载:**
```python
def load_coord(filepath):
    try:
        # [Optimization] 优先尝试二进制 npy 加载
        return np.load(filepath)
    except (ValueError, OSError, pickle.UnpicklingError):
        # [Optimization] 回退到 pandas 快速 CSV 读取
        return pd.read_csv(filepath, header=None).values
```

**genie/utils/data_io.py:**
```python
def load_coord(filepath):
    # 总是使用慢速的 loadtxt
    return np.loadtxt(filepath, delimiter=',')
```

### 11.3 缓存优化效果

| 特性 | 无缓存 | 有缓存 |
|------|--------|--------|
| 首次启动 | 慢 (需要扫描所有文件) | 慢 (但结果被缓存) |
| 后续启动 | 慢 (重复扫描) | 快 (直接加载缓存) |
| 百万级 AFDB 文件 | 几分钟 | 几秒 |
| 网络存储 | 非常慢 | 快 |

### 11.4 关键优化点

1. **Pickle 缓存**: 使用 pickle 序列化文件路径列表
2. **缓存键生成**: 基于数据集名称、min/max 残基数、类选择生成唯一键
3. **智能加载**: 优先二进制格式，失败回退到 CSV
4. **Pandas 加速**: 使用 pandas 读取 CSV 比 numpy.loadtxt 快

---

## 12. 数据集预处理脚本优化 (scripts)

### 12.1 二进制格式 vs 文本格式存储

**scripts/generate_scope_coords.py:**
```python
# 使用文本格式存储，速度慢，文件大
coords_filepath = os.path.join(coords_dir, '{}.npy'.format(domain['domain_id']))
np.savetxt(coords_filepath, coords, delimiter=',')
```

**scripts-v1/generate_scope_coords.py:**
```python
# [Optimization] 使用二进制格式存储，速度快，文件小
coords_filepath = os.path.join(coords_dir, '{}.npy'.format(domain['domain_id']))
np.save(coords_filepath, coords)
```

### 12.2 存储格式对比

| 特性 | np.savetxt (文本) | np.save (二进制) |
|------|-------------------|------------------|
| 存储速度 | 慢 | 快 (~10x) |
| 文件大小 | 大 (~3x) | 小 |
| 读取速度 | 慢 | 快 (~100x) |
| 精度 | 可能损失 | 完全保留 |

## 12. 后续优化方向，这些都有已完成代码在[Flash_genie](https://github.com/northws/Flash_genie)仓库里，但现在效果很烂还需优化。

1. **Flash IPA**: 当前已预留 `use_flash_ipa` 配置，需要实现高效的 Flash Attention IPA，
2. **梯度累积**: 支持虚拟 batch size 以处理更大序列
3. **DeepSpeed 集成**: 进一步优化分布式训练
4. **模型并行**: 支持超大模型的跨设备分割
