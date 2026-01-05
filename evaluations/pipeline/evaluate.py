import wandb
import argparse
import torch
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# 假设这些模块在你现有的工程结构中
from genie.config import Config
from genie.data.data_module import SCOPeDataModule
from genie.diffusion.genie import Genie


def main(args):
    # ==========================================
    # 1. 针对 RTX 5090 (Blackwell) 的核心硬件优化
    # ==========================================
    # 'high' 模式下，TF32 Tensor Cores 全速运行。
    # 对于深度学习训练，这通常是速度与精度的最佳平衡点。
    torch.set_float32_matmul_precision('high')

    # 加载配置
    config = Config(filename=args.config)

    # ==========================================
    # 2. 设备与加速器逻辑
    # ==========================================
    if args.gpus is not None:
        # 将输入字符串 "0,1" 转换为列表 [0, 1]
        gpus = [int(elt) for elt in args.gpus.split(',')]
        accelerator = 'gpu'
        strategy = 'ddp_find_unused_parameters_true'
    # 注意: 如果你的模型确保所有参数在每次 forward 中都参与计算，
    # 改用 'ddp' 会稍微更快一点。但 'ddp_find_unused_parameters_true' 更安全，不易报错。
    else:
        gpus = 'auto'
        accelerator = 'auto'
        strategy = 'auto'

    # ==========================================
    # 3. 日志与回调设置
    # ==========================================
    # 确保日志目录存在
    os.makedirs(config.io['log_dir'], exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=config.io['log_dir'],
        name=config.io['name']
    )

    wandb_logger = WandbLogger(
        project=config.io['name'],
        name=f"{config.io['name']}_run"
    )

    checkpoint_path = os.path.join(config.io['log_dir'], config.io['name'], "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config.training['checkpoint_every_n_epoch'],
        dirpath=checkpoint_path,
        filename='genie-{epoch:02d}-{step:04d}',  # 格式化文件名，方便排序
        save_top_k=-1  # 保存所有 checkpoint，或者根据显盘空间设置为 3
    )

    # 设置随机种子 (workers=True 确保 DataLoader 的 worker 也是确定性的)
    seed_everything(config.training['seed'], workers=True)

    # ==========================================
    # 4. 数据与模型加载
    # ==========================================
    # DataModule: 传入 batch_size 和配置中的 IO 路径
    dm = SCOPeDataModule(**config.io, batch_size=config.training['batch_size'])

    # Model: 初始化模型
    model = Genie(config)

    # ==========================================
    # 5. Trainer 配置 (性能最大化)
    # ==========================================
    trainer = Trainer(
        accelerator=accelerator,
        devices=gpus,
        logger=[tb_logger, wandb_logger],

        # 多卡并行策略
        strategy=strategy,

        # [关键优化] BF16 混合精度
        # RTX 30/40/50 系列首选。比 FP16 更稳（不易 NaN），且速度相当。
        precision='bf16-mixed',

        # [关键优化] 确定性与基准测试
        deterministic=False,  # 关闭确定性算法以提升速度
        benchmark=True,  # 开启 CuDNN Benchmark，自动寻找最适合当前硬件的卷积算法

        enable_progress_bar=True,
        log_every_n_steps=config.training['log_every_n_step'],
        max_epochs=config.training['n_epoch'],
        callbacks=[checkpoint_callback],

        # 梯度累积 (可选)：如果 5090 显存依然不够大 Batch Size，可以用这个参数
        # accumulate_grad_batches=1
    )

    # 开始训练
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 允许输入 "0,1" 来指定使用两张卡
    parser.add_argument('-g', '--gpus', type=str, default=None, help='GPU devices to use (e.g., "0,1")')
    parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
    args = parser.parse_args()

    main(args)