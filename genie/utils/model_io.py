import os
import sys
import glob
import numpy as np

from genie.config import Config
from genie.diffusion.genie import Genie


def get_versions(rootdir, name):
	basedir = os.path.join(rootdir, name)
	return sorted([
		int(version_dir.split('_')[-1]) 
		for version_dir in glob.glob(os.path.join(basedir, 'version_*'), recursive=False)
	])

def get_epochs(rootdir, name, version):
	basedir = os.path.join(rootdir, name)
	# Try both locations: under version folder (default lightning) and directly under model folder
	ckpt_paths = glob.glob(os.path.join(basedir, 'version_{}'.format(version), 'checkpoints', '*.ckpt'))
	if not ckpt_paths:
		ckpt_paths = glob.glob(os.path.join(basedir, 'checkpoints', '*.ckpt'))
	
	if not ckpt_paths:
		return []

	return sorted([
		int(epoch_filepath.split('epoch=')[-1].split('-')[0])
		for epoch_filepath in ckpt_paths
	])

def load_model(rootdir, name, version=None, epoch=None):

	# load configuration and create default model
	basedir = os.path.join(rootdir, name)
	# config_filepath = os.path.join(basedir, 'configuration') # Original
	# Config might be just a yaml file you used for training, usually not saved as 'configuration' binary unless custom
	# For now, let's assume we can re-use the training config or there is a file. 
	# BUT looking at train.py, it loads config from args and doesn't explicitly save a 'configuration' file in the run dir 
	# the standard way. However, let's look at how Genie is initialized.
    # It seems your workspace has runs/config.yaml. Let's try to find a config.
	
	config = Config(os.path.join('genie', 'runs', 'config.yaml')) # Fallback to default config path we saw earlier



	# check for latest version if needed
	available_versions = get_versions(rootdir, name)
	if version is None:
		if len(available_versions) == 0:
			print('No checkpoint available (version)')
			sys.exit(0)
		version = np.max(available_versions)
	else:
		if version not in available_versions:
			print('Missing checkpoint version: {}'.format(version))
			sys.exit(0)

	# check for latest epoch if needed
	available_epochs = get_epochs(rootdir, name, version)
	if epoch is None:
		if len(available_epochs) == 0:
			print('No checkpoint available (epoch)')
			sys.exit(0)
		epoch = np.max(available_epochs)
	else:
		if epoch not in available_epochs:
			print('Missing checkpoint epoch: {}'.format(epoch))
			print('Available epochs:', available_epochs) 
			sys.exit(0)

	# load checkpoint
	# Try looking in both places again to construct path
	ckpt_filename_pattern = 'epoch={}*.ckpt'.format(epoch)
	ckpt_filepath = None
	
	possible_paths = [
		os.path.join(basedir, 'version_{}'.format(version), 'checkpoints', ckpt_filename_pattern),
		os.path.join(basedir, 'checkpoints', ckpt_filename_pattern)
	]
	
	for path_pattern in possible_paths:
		found = glob.glob(path_pattern)
		if found:
			ckpt_filepath = found[0] # Take the first match (e.g. epoch=249-step=123250.ckpt)
			break
			
	if ckpt_filepath is None:
		print(f"Could not find checkpoint file for epoch {epoch}")
		sys.exit(1)

	print(f"Loading checkpoint from: {ckpt_filepath}")
	diffusion = Genie.load_from_checkpoint(ckpt_filepath, config=config)

	
	# save checkpoint information
	diffusion.rootdir = rootdir
	diffusion.name = name
	diffusion.version = version
	diffusion.epoch = epoch
	diffusion.checkpoint = ckpt_filepath

	return diffusion
