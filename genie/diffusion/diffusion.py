import torch
from tqdm import tqdm
from torch.optim import Adam, AdamW
from abc import ABC, abstractmethod
from pytorch_lightning.core import LightningModule

from genie.model.model import Denoiser

# Conditional import for Flash Denoiser
try:
    from genie.model.flash_denoiser import FlashDenoiser
    HAS_FLASH_DENOISER = True
except ImportError:
    HAS_FLASH_DENOISER = False


class Diffusion(LightningModule, ABC):

	def __init__(self, config):
		super(Diffusion, self).__init__()

		self.config = config
		
		# Determine whether to use Flash mode
		use_flash_mode = config.training.get('use_flash_mode', False)
		max_n_res = config.io.get('max_n_res', 256)
		
		if use_flash_mode:
			if not HAS_FLASH_DENOISER:
				print("Warning: use_flash_mode=True but flash_ipa not installed. Falling back to standard Denoiser.")
				use_flash_mode = False
			else:
				print(f"Using FlashDenoiser (memory-efficient mode) for max_n_res={max_n_res}")
		
		if use_flash_mode and HAS_FLASH_DENOISER:
			# Use memory-efficient Flash Denoiser
			# Extract only the parameters FlashDenoiser needs from config.model
			# (avoid duplicating max_n_res which is already in config.model)
			model_params = {k: v for k, v in self.config.model.items() 
			               if k not in ['max_n_res', 'use_flash_ipa', 'use_grad_checkpoint', 'z_factor_rank', 'k_neighbors', 'use_flash_attn_3']}
			self.model = FlashDenoiser(
				**model_params,
				n_timestep=self.config.diffusion['n_timestep'],
				max_n_res=max_n_res,
				z_factor_rank=config.model.get('z_factor_rank', 2),
				k_neighbors=config.model.get('k_neighbors', 10),
				use_grad_checkpoint=config.training.get('use_grad_checkpoint', False),
				use_flash_attn_3=config.model.get('use_flash_attn_3', True)
			)
		else:
			# Use standard Denoiser
			self.model = Denoiser(
				**self.config.model,
				n_timestep=self.config.diffusion['n_timestep']
			)

		self.setup = False

	@abstractmethod
	def setup_schedule(self):
		'''
		Set up variance schedule and precompute its corresponding terms.
		'''
		raise NotImplemented

	@abstractmethod
	def transform(self, batch):
		'''
		Transform batch data from data pipeline into the desired format

		Input:
			batch - coordinates from data pipeline (shape: b x (n_res * 3))

		Output: frames (shape: b x n_res)
		'''
		raise NotImplemented

	@abstractmethod
	def sample_timesteps(self, num_samples):
		raise NotImplemented

	@abstractmethod
	def sample_frames(self, mask):
		raise NotImplemented

	@abstractmethod
	def q(self, t0, s, mask):
		raise NotImplemented

	@abstractmethod
	def p(self, ts, s, mask, noise_scale):
		raise NotImplemented

	@abstractmethod
	def loss_fn(self, tnoise, ts, s):
		raise NotImplemented

	# Optimization: Set default noise_scale to 0.4 based on Genie 2 findings [cite: 110]
	# Lower noise scale (e.g. 0.4) improves designability (scTM > 0.5).
	def p_sample_loop(self, mask, noise_scale=0.4, verbose=True):
		if not self.setup:
			self.setup_schedule()
			self.setup = True
		ts = self.sample_frames(mask)
		ts_seq = [ts]
		for i in tqdm(reversed(range(self.config.diffusion['n_timestep'])), desc='sampling loop time step', total=self.config.diffusion['n_timestep'], disable=not verbose):
			s = torch.Tensor([i] * mask.shape[0]).long().to(self.device)
			ts = self.p(ts, s, mask, noise_scale)
			ts_seq.append(ts)
		return ts_seq

	def training_step(self, batch, batch_idx):
		'''
		Training iteration.

		Input:
			batch     - coordinates from data pipeline (shape: b x (n_res * 3))
			batch_idx - batch index (shape: b)

		Output: Either a single loss value or a dictionary of losses, containing
			one key as 'loss' (loss value for optimization)
		'''
		if not self.setup:
			self.setup_schedule()
			self.setup = True
		t0, mask = self.transform(batch)
		s = self.sample_timesteps(t0.shape[0])
		ts, tnoise = self.q(t0, s, mask)
		loss = self.loss_fn(tnoise, ts, s, mask)
		self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
		return loss

	def configure_optimizers(self):
		# Optimization: Use Fused AdamW if available on GPU
		# Note: Gradient Clipping must be disabled in Trainer for this to work
		use_fused = torch.cuda.is_available()
		
		# Large batch training: Scale learning rate
		# Linear scaling rule: lr_new = lr_base * (batch_size / base_batch_size)
		base_lr = self.config.optimization['lr']
		base_batch_size = self.config.optimization.get('base_batch_size', 8)
		actual_batch_size = self.config.training['batch_size']
		lr_scale = self.config.training.get('lr_scale_factor', 1.0)
		
		# Apply square root scaling (more stable than linear for large batches)
		if lr_scale == 1.0 and actual_batch_size > base_batch_size:
			lr_scale = (actual_batch_size / base_batch_size) ** 0.5
			print(f"[Large Batch] Auto LR scaling: {base_lr:.2e} -> {base_lr * lr_scale:.2e} (sqrt rule)")
		
		scaled_lr = base_lr * lr_scale
		
		optimizer = AdamW(
			self.model.parameters(),
			lr=scaled_lr,
			fused=use_fused
		)
		
		# Add warmup scheduler for large batch training
		warmup_epochs = self.config.training.get('warmup_epochs', 0)
		if warmup_epochs > 0:
			print(f"[Large Batch] Using {warmup_epochs} warmup epochs")
			from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
			
			# Warmup from 10% to 100% of scaled_lr
			warmup_scheduler = LinearLR(
				optimizer, 
				start_factor=0.1, 
				end_factor=1.0, 
				total_iters=warmup_epochs
			)
			
			# Optional: Cosine decay after warmup
			total_epochs = self.config.training['n_epoch']
			eta_min_factor = self.config.training.get('cosine_eta_min_factor', 0.01)
			cosine_scheduler = CosineAnnealingLR(
				optimizer,
				T_max=total_epochs - warmup_epochs,
				eta_min=scaled_lr * eta_min_factor
			)
			
			print(f"[LR Schedule] Warmup: {scaled_lr * 0.1:.2e} -> {scaled_lr:.2e} ({warmup_epochs} epochs)")
			print(f"[LR Schedule] Cosine decay: {scaled_lr:.2e} -> {scaled_lr * eta_min_factor:.2e} ({total_epochs - warmup_epochs} epochs)")
			
			scheduler = SequentialLR(
				optimizer,
				schedulers=[warmup_scheduler, cosine_scheduler],
				milestones=[warmup_epochs]
			)
			
			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': scheduler,
					'interval': 'epoch',
					'frequency': 1
				}
			}
		
		return optimizer