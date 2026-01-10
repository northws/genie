import os
import sys

# Add the project root to sys.path to enable imports from the 'genie' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm, trange

from genie.utils.model_io import load_model



def main(args):

	# device
	device = 'cuda:{}'.format(args.gpu) if args.gpu is not None else 'cpu'

	# model
	model = load_model(args.rootdir, args.model_name, args.model_version, args.model_epoch).to(device)

	# output directory
	outdir = os.path.join(model.rootdir, model.name, 'version_{}'.format(model.version), 'samples')
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	outdir = os.path.join(outdir, 'epoch_{}'.format(model.epoch))
	if os.path.exists(outdir):
		print('Samples existed!')
		# sys.exit(0) # Removed exit to allow overwriting or adding to existing samples
	else:
		os.mkdir(outdir)


	# sanity check
	min_length = args.min_length
	max_length = args.max_length
	max_n_res = model.config.io['max_n_res']
	assert max_length <= max_n_res

	# sample
	for length in trange(min_length, max_length + 1):
		for batch_idx in range(args.num_batches):
			mask = torch.cat([
				torch.ones((args.batch_size, length)),
				torch.zeros((args.batch_size, max_n_res - length))
			], dim=1).to(device)
			ts_seq = model.p_sample_loop(mask, args.noise_scale, verbose=True)
			ts = ts_seq[-1]
			for batch_sample_idx in range(ts.shape[0]):
				sample_idx = batch_idx * args.batch_size + batch_sample_idx
				coords = ts[batch_sample_idx].trans.detach().cpu().numpy()
				coords = coords[:length]
				np.savetxt(os.path.join(outdir, f'{length}_{sample_idx}.npy'), coords, fmt='%.3f', delimiter=',')
				
				if args.save_trajectory:
					# Save trajectory for this sample
					traj_coords = []
					for step_ts in ts_seq:
						step_coords = step_ts[batch_sample_idx].trans.detach().cpu().numpy()
						step_coords = step_coords[:length]
						# Center each frame for visualization stability
						step_coords = step_coords - step_coords.mean(axis=0)
						traj_coords.append(step_coords)
					traj_coords = np.array(traj_coords) # [Steps, Length, 3]
					np.save(os.path.join(outdir, f'{length}_{sample_idx}_traj.npy'), traj_coords)


if __name__ == '__main__':

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type=str, nargs='?', const='0', help='GPU device to use')
	parser.add_argument('-r', '--rootdir', type=str, help='Root directory (default to runs)', default='runs')
	parser.add_argument('-n', '--model_name', type=str, help='Name of Genie model', required=True)
	parser.add_argument('-v', '--model_version', type=int, help='Version of Genie model')
	parser.add_argument('-e', '--model_epoch', type=int, help='Epoch Genie model checkpointed')
	parser.add_argument('--batch_size', type=int, help='Batch size', default=5)
	parser.add_argument('--num_batches', type=int, help='Number of batches', default=2)
	parser.add_argument('--noise_scale', type=float, help='Sampling noise scale', default=0.6)
	parser.add_argument('--min_length', type=int, help='Minimum length', default=50)
	parser.add_argument('--max_length', type=int, help='Maximum length', default=128)
	parser.add_argument('--save_trajectory', action='store_true', help='Save all timesteps for visualization')
	args = parser.parse_args()

	# run
	try:
		main(args)
	except RuntimeError as e:
		if 'out of memory' in str(e).lower():
			print('\n' + '='*60)
			print('CRITICAL ERROR: CUDA Out of Memory (OOM) during sampling.')
			print('='*60)
			if torch.cuda.is_available():
				print(f'Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
				print(f'Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB')
			print('Suggestions:')
			print('1. Reduce --batch_size')
			print('2. Reduce --max_length')
			print('='*60 + '\n')
			sys.exit(1)
		else:
			raise e