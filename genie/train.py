import wandb
import argparse
import os

# [Optimization] Reduce memory fragmentation
# Setting this before importing torch to ensure it takes effect
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch  # [Added] Needed for matmul precision settings

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from genie.config import Config
from genie.data.data_module import SCOPeDataModule
from genie.diffusion.genie import Genie


def main(args):
    # [Optimization] Enable TF32 on Ampere+ GPUs (A100, RTX3090, etc.)
    # 'medium' or 'high' enables Tensor Cores for float32 matrix multiplications
    torch.set_float32_matmul_precision('medium')

    # [Optimization] Enable cuDNN benchmark for fixed input sizes
    # This finds the best convolution algorithms for the hardware
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # configuration
    config = Config(filename=args.config)

    # devices logic
    if args.gpus is not None:
        gpus = [int(elt) for elt in args.gpus.split(',')]
        accelerator = 'gpu'
    else:
        gpus = 'auto'
        accelerator = 'auto'

    # logger
    tb_logger = TensorBoardLogger(
        save_dir=config.io['log_dir'],
        name=config.io['name']
    )
    wandb_logger = WandbLogger(project=config.io['name'], name=config.io['name'])

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config.training['checkpoint_every_n_epoch'],
        dirpath=f"{config.io['log_dir']}/{config.io['name']}/checkpoints",  # Explicit path often helps
        filename='{epoch}-{step}',
        save_top_k=-1
    )

    # seed
    seed_everything(config.training['seed'], workers=True)

    # data module
    # Passing config.io as kwargs, assuming it contains data paths
    dm = SCOPeDataModule(**config.io, batch_size=config.training['batch_size'], num_workers=config.training['num_workers'])

    # model
    model = Genie(config)

    # trainer
    trainer = Trainer(
        accelerator=accelerator,  # [Updated] Explicit accelerator definition
        devices=gpus,  # [Updated] Replaces the old 'gpus' arg
        logger=[tb_logger, wandb_logger],
        strategy='ddp',
        # 'ddp' is fine, usually explicitly handling unused params is safer for complex models

        # [Optimization] Tensor Core Support
        # '16-mixed' uses FP16 for matmul (Tensor Cores) and FP32 for stability.
        # Use 'bf16-mixed' if you are on A100/H100 for better stability.
        precision='bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed',
        
        # [Stability] Gradient Clipping to prevent NaN loss
        gradient_clip_val=1.0,

        deterministic=False,  # [Optimization] Changed to False for speed unless reproducibility is strictly required
        enable_progress_bar=True,  # Changed to True usually for UX, set False if running in strict pipeline
        log_every_n_steps=config.training['log_every_n_step'],
        max_epochs=config.training['n_epoch'],
        callbacks=[checkpoint_callback]
    )

    # run
    trainer.fit(model, dm)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', type=str, help='GPU devices to use (e.g., "0,1")')
    parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
    args = parser.parse_args()

    # run
    main(args)