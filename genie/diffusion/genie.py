import torch

from genie.diffusion.diffusion import Diffusion
from genie.diffusion.schedule import get_betas
from genie.utils.loss import rmsd
from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames

# Import mHC loss regularization
try:
    from genie.diffusion.mhc_loss import compute_mhc_regularization, AdvancedMHCLoss
    HAS_MHC_LOSS = True
except ImportError:
    HAS_MHC_LOSS = False


class Genie(Diffusion):

	def setup_schedule(self):

		self.betas = get_betas(self.config.diffusion['n_timestep'], self.config.diffusion['schedule']).to(self.device)
		self.alphas = 1. - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, 0)
		self.alphas_cumprod_prev = torch.cat([torch.Tensor([1.]).to(self.device), self.alphas_cumprod[:-1]])
		self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod
		self.one_minus_alphas_cumprod_prev = 1. - self.alphas_cumprod_prev
		
		self.sqrt_betas = torch.sqrt(self.betas)
		self.sqrt_alphas = torch.sqrt(self.alphas)
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = 1. / self.sqrt_alphas_cumprod
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

		self.posterior_mean_coef1 = self.betas * self.sqrt_alphas_cumprod_prev / self.one_minus_alphas_cumprod
		self.posterior_mean_coef2 = self.one_minus_alphas_cumprod_prev * self.sqrt_alphas / self.one_minus_alphas_cumprod
		self.posterior_variance = self.betas * self.one_minus_alphas_cumprod_prev / self.one_minus_alphas_cumprod

	def transform(self, batch):

		coords, mask = batch
		coords = coords.float()
		mask = mask.float()

		ca_coords = coords[:, 1::3]
		trans = ca_coords - torch.mean(ca_coords, dim=1, keepdim=True)
		rots = compute_frenet_frames(trans, mask)

		return T(rots, trans), mask

	def sample_timesteps(self, num_samples):
		return torch.randint(0, self.config.diffusion['n_timestep'], size=(num_samples,)).to(self.device)

	def sample_frames(self, mask):
		trans = torch.randn((mask.shape[0], mask.shape[1], 3)).to(self.device)
		trans = trans * mask.unsqueeze(-1)
		rots = compute_frenet_frames(trans, mask)
		return T(rots, trans)

	def q(self, t0, s, mask):

		# [b, n_res, 3]
		trans_noise = torch.randn_like(t0.trans) * mask.unsqueeze(-1)
		rots_noise = torch.eye(3).view(1, 1, 3, 3).repeat(t0.shape[0], t0.shape[1], 1, 1).to(self.device)

		trans = self.sqrt_alphas_cumprod[s].view(-1, 1, 1).to(self.device) * t0.trans + \
			self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1).to(self.device) * trans_noise
		rots = compute_frenet_frames(trans, mask)

		return T(rots, trans), T(rots_noise, trans_noise)

	def p(self, ts, s, mask, noise_scale):

		# [b, 1, 1]
		w_noise = ((1. - self.alphas[s].to(self.device)) / self.sqrt_one_minus_alphas_cumprod[s].to(self.device)).view(-1, 1, 1)

		# [b, n_res]
		noise_pred_trans = ts.trans - self.model(ts, s, mask).trans
		noise_pred_rots = torch.eye(3).view(1, 1, 3, 3).repeat(ts.shape[0], ts.shape[1], 1, 1)
		noise_pred = T(noise_pred_rots, noise_pred_trans)

		# [b, n_res, 3]
		trans_mean = (1. / self.sqrt_alphas[s]).view(-1, 1, 1).to(self.device) * (ts.trans - w_noise * noise_pred.trans)
		trans_mean = trans_mean * mask.unsqueeze(-1)

		if (s == 0.0).all():
			rots_mean = compute_frenet_frames(trans_mean, mask)
			return T(rots_mean.detach(), trans_mean.detach())
		else:

			# [b, n_res, 3]
			trans_z = torch.randn_like(ts.trans).to(self.device)

			# [b, 1, 1]
			trans_sigma = self.sqrt_betas[s].view(-1, 1, 1).to(self.device)

			# [b, n_res, 3]
			trans = trans_mean + noise_scale * trans_sigma * trans_z
			trans = trans * mask.unsqueeze(-1)

			# [b, n_res, 3, 3]
			rots = compute_frenet_frames(trans, mask)

			return T(rots.detach(), trans.detach())

	def loss_fn(self, tnoise, ts, s, mask):

		noise_pred_trans = ts.trans - self.model(ts, s, mask).trans

		trans_loss = rmsd(
			noise_pred_trans,
			tnoise.trans,
			mask
		)

		# Get training config
		use_mhc_loss = False
		use_adv_mhc_loss = False
		mhc_weight = 0.01
		if hasattr(self.config, 'training'):
			if isinstance(self.config.training, dict):
				use_mhc_loss = self.config.training.get('use_mhc_loss', False)
				use_adv_mhc_loss = self.config.training.get('use_adv_mhc_loss', False)
				mhc_weight = self.config.training.get('mhc_loss_weight', 0.01)
			else:
				use_mhc_loss = getattr(self.config.training, 'use_mhc_loss', False)
				use_adv_mhc_loss = getattr(self.config.training, 'use_adv_mhc_loss', False)
				mhc_weight = getattr(self.config.training, 'mhc_loss_weight', 0.01)
		
		# Advanced mHC Loss: 使用所有 mHC 正则化函数
		if use_adv_mhc_loss and HAS_MHC_LOSS:
			# 从配置中获取高级 mHC 损失的权重
			if isinstance(self.config.training, dict):
				balance_weight = self.config.training.get('adv_mhc_balance_weight', 0.1)
				norm_weight = self.config.training.get('adv_mhc_norm_weight', 0.1)
				stability_weight = self.config.training.get('adv_mhc_stability_weight', 0.05)
				flow_weight = self.config.training.get('adv_mhc_flow_weight', 0.01)
				ds_weight = self.config.training.get('adv_mhc_ds_weight', 0.0)
				target_ratio = self.config.training.get('adv_mhc_target_ratio', 0.5)
				# 长序列训练参数
				long_seq_mode = self.config.training.get('adv_mhc_long_seq_mode', False)
				# 如果使用 Flash 模式，自动启用长序列模式
				use_flash_mode = self.config.training.get('use_flash_mode', False)
				if use_flash_mode and not long_seq_mode:
					long_seq_mode = True  # Flash 模式下自动启用长序列稳定化
				adaptive_norm_weight = self.config.training.get('adv_mhc_adaptive_norm_weight', 0.05)
				magnitude_clip_weight = self.config.training.get('adv_mhc_magnitude_clip_weight', 0.02)
				local_consistency_weight = self.config.training.get('adv_mhc_local_consistency_weight', 0.01)
				spectral_norm_weight = self.config.training.get('adv_mhc_spectral_norm_weight', 0.02)
				base_seq_len = self.config.training.get('adv_mhc_base_seq_len', 128)
				max_magnitude = self.config.training.get('adv_mhc_max_magnitude', 10.0)
				consistency_window = self.config.training.get('adv_mhc_consistency_window', 5)
			else:
				balance_weight = getattr(self.config.training, 'adv_mhc_balance_weight', 0.1)
				norm_weight = getattr(self.config.training, 'adv_mhc_norm_weight', 0.1)
				stability_weight = getattr(self.config.training, 'adv_mhc_stability_weight', 0.05)
				flow_weight = getattr(self.config.training, 'adv_mhc_flow_weight', 0.01)
				ds_weight = getattr(self.config.training, 'adv_mhc_ds_weight', 0.0)
				target_ratio = getattr(self.config.training, 'adv_mhc_target_ratio', 0.5)
				# 长序列训练参数
				long_seq_mode = getattr(self.config.training, 'adv_mhc_long_seq_mode', False)
				use_flash_mode = getattr(self.config.training, 'use_flash_mode', False)
				if use_flash_mode and not long_seq_mode:
					long_seq_mode = True
				adaptive_norm_weight = getattr(self.config.training, 'adv_mhc_adaptive_norm_weight', 0.05)
				magnitude_clip_weight = getattr(self.config.training, 'adv_mhc_magnitude_clip_weight', 0.02)
				local_consistency_weight = getattr(self.config.training, 'adv_mhc_local_consistency_weight', 0.01)
				spectral_norm_weight = getattr(self.config.training, 'adv_mhc_spectral_norm_weight', 0.02)
				base_seq_len = getattr(self.config.training, 'adv_mhc_base_seq_len', 128)
				max_magnitude = getattr(self.config.training, 'adv_mhc_max_magnitude', 10.0)
				consistency_window = getattr(self.config.training, 'adv_mhc_consistency_window', 5)
			
			# 创建或复用 AdvancedMHCLoss 实例
			if not hasattr(self, '_adv_mhc_loss'):
				self._adv_mhc_loss = AdvancedMHCLoss(
					balance_weight=balance_weight,
					norm_weight=norm_weight,
					stability_weight=stability_weight,
					flow_weight=flow_weight,
					ds_penalty_weight=ds_weight,
					base_reg_weight=mhc_weight,
					target_residual_ratio=target_ratio,
					# 长序列训练参数
					long_seq_mode=long_seq_mode,
					adaptive_norm_weight=adaptive_norm_weight,
					magnitude_clip_weight=magnitude_clip_weight,
					local_consistency_weight=local_consistency_weight,
					spectral_norm_weight=spectral_norm_weight,
					base_seq_len=base_seq_len,
					max_magnitude=max_magnitude,
					consistency_window=consistency_window,
				)
				if long_seq_mode:
					print(f"[AdvMHCLoss] Long sequence mode enabled for gradient stabilization")
			
			# 使用完整版计算（包含所有 mHC 损失函数）
			adv_loss_dict = self._adv_mhc_loss.compute_diffusion_loss(
				noise_pred=noise_pred_trans,
				noise_target=tnoise.trans,
				mask=mask,
			)
			
			# 记录各项损失
			self.log('train/rmsd_loss', trans_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
			self.log('train/adv_mhc_total', adv_loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			
			# 基础 mHC 损失（6 个核心函数）
			self.log('train/adv_mhc_base_reg', adv_loss_dict['base_reg_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			self.log('train/adv_mhc_norm', adv_loss_dict['norm_preservation_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			self.log('train/adv_mhc_flow', adv_loss_dict['flow_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			self.log('train/adv_mhc_balance', adv_loss_dict['balance_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			self.log('train/adv_mhc_stability', adv_loss_dict['stability_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			self.log('train/adv_mhc_ds_penalty', adv_loss_dict['ds_penalty_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			
			# 长序列训练损失日志（4 个额外函数）
			if long_seq_mode:
				self.log('train/adv_mhc_adaptive_norm', adv_loss_dict['adaptive_norm_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
				self.log('train/adv_mhc_magnitude_clip', adv_loss_dict['magnitude_clip_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
				self.log('train/adv_mhc_local_consistency', adv_loss_dict['local_consistency_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
				self.log('train/adv_mhc_spectral_norm', adv_loss_dict['spectral_norm_loss'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			
			total_loss = trans_loss + adv_loss_dict['total_loss']
			return total_loss
		
		# Standard mHC Loss (简化版)
		elif use_mhc_loss and HAS_MHC_LOSS:
			mhc_reg = compute_mhc_regularization(
				noise_pred_trans,
				tnoise.trans,
				mask,
				weight=mhc_weight
			)
			
			# 分开记录，方便比较
			# trans_loss: 主损失（可与不使用 mHC 的实验比较）
			# mhc_reg: 正则化项（额外的稳定性损失）
			# total_loss: 用于反向传播
			self.log('train/rmsd_loss', trans_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
			self.log('train/mhc_reg', mhc_reg, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
			
			total_loss = trans_loss + mhc_reg
			return total_loss
		
		return trans_loss