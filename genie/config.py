import os

int_or_none = lambda x: int(x) if x is not None else None
str_list_or_none = lambda x: x.strip().split(',') if x is not None else None
int_list_or_none = lambda x: int(x.strip().split(',')) if x is not None else None
eval_if_str = lambda x: literal_eval(x) if isinstance(x, str) else x

class Config:

	def __init__(self, filename=None):
		config = {} if filename is None else self._load_config(filename)
		self._create_config(config)

	def _create_config(self, config):

		self.io = {
			'name':                             config.get('name',               None),
			'max_n_res':            int_or_none(config.get('maximumNumResidues', None)),
			'min_n_res':            int_or_none(config.get('minimumNumResidues', None)),
			'log_dir':                          config.get('logDirectory',       'runs'),
			'data_dir':                         config.get('dataDirectory',      'data'),
			'dataset_names':   str_list_or_none(config.get('datasetNames',       'scope')),
			'dataset_size':         int_or_none(config.get('datasetSize',        None)),
			'dataset_classes': str_list_or_none(config.get('datasetClasses',     None))
		}

		self.diffusion = {
			'n_timestep': int(config.get('numTimesteps', 1000)),
			'schedule':       config.get('schedule',     'cosine'),
		}

		self.model = {

			# general
			'c_s':                            int(config.get('singleFeatureDimension',                  128)),
			'c_p':                            int(config.get('pairFeatureDimension',                    128)),

			# single feature network
			'c_pos_emb':                      int(config.get('positionalEmbeddingDimension',            128)),
			'c_timestep_emb':                 int(config.get('timestepEmbeddingDimension',              128)),

			# pair feature network
			'relpos_k':                       int(config.get('relativePositionK',                       32)),
			'template_type':                      config.get('templateType',                            'v1'),

			# pair transform network
			'n_pair_transform_layer':         int(config.get('numPairTransformLayers',                  5)),
			'include_mul_update':                 config.get('includeTriangularMultiplicativeUpdate',   True),
			'include_tri_att':                    config.get('includeTriangularAttention',              False),
			'c_hidden_mul':                   int(config.get('triangularMultiplicativeHiddenDimension', 128)),
			'c_hidden_tri_att':               int(config.get('triangularAttentionHiddenDimension',      32)),
			'n_head_tri':                     int(config.get('triangularAttentionNumHeads',             4)),
			'tri_dropout':                  float(config.get('triangularDropout',                       0.25)),
			'pair_transition_n':              int(config.get('pairTransitionN',                         4)),

			# structure network
			'n_structure_layer':              int(config.get('numStructureLayers',                      5)),
			'n_structure_block':              int(config.get('numStructureBlocks',                      1)),
			'c_hidden_ipa':                   int(config.get('ipaHiddenDimension',                      16)),
			'n_head_ipa':                     int(config.get('ipaNumHeads',                             12)),
			'n_qk_point':                     int(config.get('ipaNumQkPoints',                          4)),
			'n_v_point':                      int(config.get('ipaNumVPoints',                           8)),
			'ipa_dropout':                  float(config.get('ipaDropout',                              0.1)),
			'n_structure_transition_layer':   int(config.get('numStructureTransitionLayers',            1)),
			'structure_transition_dropout': float(config.get('structureTransitionDropout',              0.1)),
			'use_flash_ipa':                      config.get('useFlashIPA',                             True),
			'max_n_res':                          self.io['max_n_res'],
			'use_grad_checkpoint':                config.get('useGradientCheckpointing',                False), # Optimization: Default to False for speed
			
			# Flash IPA specific parameters (only used when useFlashMode=True)
			'z_factor_rank':                  int(config.get('zFactorRank',                             2)),
			'k_neighbors':                    int(config.get('kNeighbors',                              10)),
			'use_flash_attn_3':                   config.get('useFlashAttn3',                           True),  # Use FA3 on Hopper GPUs if available
		}

		self.training = {
			'seed':                     int(config.get('seed',                   100)),
			'n_epoch':                  int(config.get('numEpoches',             1)),
			'batch_size':               int(config.get('batchSize',              32)),
			'num_workers':              int(config.get('numWorkers',             4)),
			'log_every_n_step':         int(config.get('logEverySteps',          1000)),
			'checkpoint_every_n_epoch': int(config.get('checkpointEveryEpoches', 500)),
			'use_grad_checkpoint':          config.get('useGradientCheckpointing', False),
			'use_flash_mode':               config.get('useFlashMode',             False),  # Memory-efficient Flash IPA mode
			'use_mhc_mode':                 config.get('useMHCMode',               False),  # mHC (Manifold-Constrained Hyper-Connections) mode
			'use_mhc_loss':                 config.get('useMHCLoss',               False),  # mHC as loss regularization only (no arch change)
			'mhc_loss_weight':        float(config.get('mhcLossWeight',           0.01)),   # Weight for mHC loss regularization
			'use_adv_mhc_loss':             config.get('useAdvMHCLoss',            False),  # Advanced mHC loss with all regularization functions
			# Advanced mHC loss hyperparameters (only used when useAdvMHCLoss=True)
			'adv_mhc_balance_weight':   float(config.get('advMHCBalanceWeight',    0.1)),   # Weight for residual balance loss
			'adv_mhc_norm_weight':      float(config.get('advMHCNormWeight',       0.1)),   # Weight for gradient norm preservation loss
			'adv_mhc_stability_weight': float(config.get('advMHCStabilityWeight',  0.05)),  # Weight for representation stability loss
			'adv_mhc_flow_weight':      float(config.get('advMHCFlowWeight',       0.01)),  # Weight for gradient flow loss
			'adv_mhc_ds_weight':        float(config.get('advMHCDoublyStochasticWeight', 0.0)),  # Weight for doubly stochastic penalty (0 to disable)
			'adv_mhc_target_ratio':     float(config.get('advMHCTargetRatio',      0.5)),   # Target residual ratio for balance loss
			# Long sequence training parameters (for Flash mode + AdvMHCLoss)
			'adv_mhc_long_seq_mode':          config.get('advMHCLongSeqMode',      False),  # Enable long sequence gradient stabilization
			'adv_mhc_adaptive_norm_weight': float(config.get('advMHCAdaptiveNormWeight', 0.05)),  # Sequence-length adaptive norm loss
			'adv_mhc_magnitude_clip_weight': float(config.get('advMHCMagnitudeClipWeight', 0.02)),  # Gradient magnitude soft clipping
			'adv_mhc_local_consistency_weight': float(config.get('advMHCLocalConsistencyWeight', 0.01)),  # Local consistency loss
			'adv_mhc_spectral_norm_weight': float(config.get('advMHCSpectralNormWeight', 0.02)),  # Spectral norm regularization
			'adv_mhc_base_seq_len':       int(config.get('advMHCBaseSeqLen',       128)),   # Base sequence length for adaptive scaling
			'adv_mhc_max_magnitude':    float(config.get('advMHCMaxMagnitude',     10.0)),  # Max prediction magnitude before penalty
			'adv_mhc_consistency_window': int(config.get('advMHCConsistencyWindow', 5)),    # Window size for local consistency
			# Large batch training optimizations
			'accumulate_grad_batches':  int(config.get('accumulateGradBatches',  1)),      # Gradient accumulation steps
			'warmup_epochs':            int(config.get('warmupEpochs',           0)),      # LR warmup epochs
			'lr_scale_factor':        float(config.get('lrScaleFactor',          1.0)),    # LR scaling for large batch
			'cosine_eta_min_factor':  float(config.get('cosineEtaMinFactor',     0.01)),   # Cosine annealing min LR factor (default 1%)
			'gradient_clip_val':      float(config.get('gradientClipVal',        1.0)),    # Gradient clipping value (None to disable)
			# mHC specific parameters
			'mhc_expansion_rate':       int(config.get('mhcExpansionRate',       4)),      # Residual stream width expansion
			'mhc_sinkhorn_iters':       int(config.get('mhcSinkhornIters',       20)),     # Sinkhorn-Knopp iterations
			'mhc_alpha_init':         float(config.get('mhcAlphaInit',           0.01)),   # Gating factor initialization
		}

		self.optimization = {
			'lr': float(config.get('learningRate', 1e-4)),
			'base_batch_size':      int(config.get('baseBatchSize',           8)),      # Reference batch size for LR scaling
		}

	def _load_config(self, filename):
		config = {}
		with open(filename) as file:
			for line in file:
				elts = line.split()
				if len(elts) == 2:
					if elts[1] == 'True':
						config[elts[0]] = True
					elif elts[1] == 'False':
						config[elts[0]] = False
					else:
						config[elts[0]] = elts[1]
		return config