"""
Flash Sample - Memory-Efficient Sampling for Long Sequences

This module provides a memory-efficient sampling script that uses Flash IPA
for generating protein backbone structures from long sequences.

Key features:
- Supports both standard and Flash mode models
- Memory-efficient for sequences > 512 residues
- Compatible with existing model checkpoints
"""

import os
import sys

# Add the project root to sys.path to enable imports from the 'genie' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm, trange

from genie.config import Config
from genie.diffusion.genie import Genie
from genie.utils.model_io import get_versions, get_epochs


def load_flash_model(rootdir, name, version=None, epoch=None, force_flash=False):
    """
    Load a Genie model with optional Flash mode override.
    
    Args:
        rootdir: Root directory containing model checkpoints
        name: Model name
        version: Model version (None for latest)
        epoch: Model epoch (None for latest)
        force_flash: If True, force Flash mode even if not specified in config
    
    Returns:
        Genie model instance
    """
    import glob
    
    # Load configuration
    basedir = os.path.join(rootdir, name)
    config_filepath = os.path.join(basedir, 'configuration')
    config = Config(config_filepath)
    
    # Override Flash mode if requested
    if force_flash:
        print("Force enabling Flash mode for sampling...")
        config.training['use_flash_mode'] = True
        # Ensure Flash-specific parameters are set
        if 'z_factor_rank' not in config.model:
            config.model['z_factor_rank'] = 2
        if 'k_neighbors' not in config.model:
            config.model['k_neighbors'] = 10
    
    # Check for latest version if needed
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
    
    # Check for latest epoch if needed
    available_epochs = get_epochs(rootdir, name, version)
    if epoch is None:
        if len(available_epochs) == 0:
            print('No checkpoint available (epoch)')
            sys.exit(0)
        epoch = np.max(available_epochs)
    else:
        if epoch not in available_epochs:
            print('Missing checkpoint epoch: {}'.format(epoch))
            print('Available epochs: {}'.format(available_epochs))
            sys.exit(0)
    
    # Find checkpoint file
    ckpt_filename_pattern = 'epoch={}*.ckpt'.format(epoch)
    ckpt_filepath = None
    
    possible_paths = [
        os.path.join(basedir, 'version_{}'.format(version), 'checkpoints', ckpt_filename_pattern),
        os.path.join(basedir, 'version_{}'.format(version), ckpt_filename_pattern),
        os.path.join(basedir, 'checkpoints', ckpt_filename_pattern)
    ]
    
    for path_pattern in possible_paths:
        found = glob.glob(path_pattern)
        if found:
            ckpt_filepath = found[0]
            break
    
    if ckpt_filepath is None:
        print(f"Could not find checkpoint file for epoch {epoch}")
        sys.exit(1)
    
    print(f"Loading checkpoint from: {ckpt_filepath}")
    
    # Load model
    # Note: When force_flash=True and the checkpoint was trained with standard mode,
    # we need to handle weight mapping carefully
    if force_flash and not config.training.get('use_flash_mode', False):
        print("Warning: Loading standard model weights into Flash model architecture.")
        print("         Some weights (PairTransformNet) will be randomly initialized.")
        print("         For best results, train a model with useFlashMode=True")
    
    diffusion = Genie.load_from_checkpoint(ckpt_filepath, config=config, strict=False)
    
    # Save checkpoint information
    diffusion.rootdir = rootdir
    diffusion.name = name
    diffusion.version = version
    diffusion.epoch = epoch
    diffusion.checkpoint = ckpt_filepath
    
    return diffusion


def sample_with_flash(model, mask, noise_scale=0.4, verbose=True):
    """
    Sample using the model with memory-efficient settings.
    
    This function wraps p_sample_loop with additional memory optimizations
    suitable for long sequences.
    
    Args:
        model: Genie model instance
        mask: Sequence mask [B, L]
        noise_scale: Sampling noise scale
        verbose: Show progress bar
    
    Returns:
        List of T objects representing the diffusion trajectory
    """
    # Enable inference mode optimizations
    with torch.inference_mode():
        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run sampling
        ts_seq = model.p_sample_loop(mask, noise_scale, verbose=verbose)
        
    return ts_seq


def main(args):
    """Main sampling function with Flash mode support."""
    
    # Device setup
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
        # Set memory allocation strategy for long sequences
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except RuntimeError:
                pass
    else:
        device = 'cpu'
    
    print("="*60)
    print("Flash Sample - Memory-Efficient Protein Generation")
    print("="*60)
    
    # Load model
    model = load_flash_model(
        args.rootdir, 
        args.model_name, 
        args.model_version, 
        args.model_epoch,
        force_flash=args.flash_mode
    ).to(device)
    
    # Check if Flash mode is active
    is_flash_mode = hasattr(model.model, '__class__') and 'Flash' in model.model.__class__.__name__
    print(f"Model type: {model.model.__class__.__name__}")
    print(f"Flash mode: {'Enabled' if is_flash_mode else 'Disabled'}")
    
    # Output directory
    outdir = os.path.join(model.rootdir, model.name, f'version_{model.version}', 'samples')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir = os.path.join(outdir, f'epoch_{model.epoch}')
    if os.path.exists(outdir):
        print(f'Output directory exists: {outdir}')
    else:
        os.makedirs(outdir)
    
    # Length validation
    min_length = args.min_length
    max_length = args.max_length
    max_n_res = model.config.io['max_n_res']
    
    if max_length > max_n_res:
        print(f"Warning: max_length ({max_length}) > max_n_res ({max_n_res})")
        print(f"         Clamping to {max_n_res}")
        max_length = max_n_res
    
    print(f"Generating lengths: {min_length} to {max_length}")
    print(f"Batch size: {args.batch_size}, Num batches: {args.num_batches}")
    print(f"Noise scale: {args.noise_scale}")
    print("="*60)
    
    # Memory estimation for Flash mode
    if is_flash_mode and torch.cuda.is_available():
        # Estimate memory usage
        estimated_mem = args.batch_size * max_length * 128 * 4 / (1024**3)  # Rough estimate in GB
        available_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Estimated peak memory: ~{estimated_mem:.1f} GB")
        print(f"Available GPU memory: {available_mem:.1f} GB")
    
    # Sample
    model.eval()
    total_samples = 0
    
    for length in trange(min_length, max_length + 1, desc="Length"):
        for batch_idx in range(args.num_batches):
            # Create mask
            mask = torch.cat([
                torch.ones((args.batch_size, length)),
                torch.zeros((args.batch_size, max_n_res - length))
            ], dim=1).to(device)
            
            # Sample with memory optimization
            ts_seq = sample_with_flash(model, mask, args.noise_scale, verbose=False)
            ts = ts_seq[-1]
            
            # Save samples
            for batch_sample_idx in range(ts.shape[0]):
                sample_idx = batch_idx * args.batch_size + batch_sample_idx
                coords = ts[batch_sample_idx].trans.detach().cpu().numpy()
                coords = coords[:length]
                
                # Save coordinates
                np.savetxt(
                    os.path.join(outdir, f'{length}_{sample_idx}.npy'), 
                    coords, fmt='%.3f', delimiter=','
                )
                
                # Save trajectory if requested
                if args.save_trajectory:
                    traj_coords = []
                    for step_ts in ts_seq:
                        step_coords = step_ts[batch_sample_idx].trans.detach().cpu().numpy()
                        step_coords = step_coords[:length]
                        step_coords = step_coords - step_coords.mean(axis=0)
                        traj_coords.append(step_coords)
                    traj_coords = np.array(traj_coords)
                    np.save(
                        os.path.join(outdir, f'{length}_{sample_idx}_traj.npy'), 
                        traj_coords
                    )
                
                total_samples += 1
            
            # Clear cache periodically for long runs
            if torch.cuda.is_available() and batch_idx % 5 == 0:
                torch.cuda.empty_cache()
    
    print("="*60)
    print(f"Sampling complete! Generated {total_samples} samples")
    print(f"Output directory: {outdir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Flash Sample - Memory-efficient protein backbone generation"
    )
    
    # Model arguments
    parser.add_argument('-g', '--gpu', type=str, nargs='?', const='0',
                        help='GPU device to use (default: None for CPU)')
    parser.add_argument('-r', '--rootdir', type=str, default='runs',
                        help='Root directory containing models')
    parser.add_argument('-n', '--model_name', type=str, required=True,
                        help='Name of Genie model')
    parser.add_argument('-v', '--model_version', type=int,
                        help='Version of Genie model (default: latest)')
    parser.add_argument('-e', '--model_epoch', type=int,
                        help='Epoch of checkpoint (default: latest)')
    
    # Sampling arguments
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for sampling')
    parser.add_argument('--num_batches', type=int, default=2,
                        help='Number of batches per length')
    parser.add_argument('--noise_scale', type=float, default=0.6,
                        help='Sampling noise scale (lower = more deterministic)')
    parser.add_argument('--min_length', type=int, default=50,
                        help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    
    # Flash mode arguments
    parser.add_argument('--flash_mode', action='store_true',
                        help='Force Flash mode for memory-efficient sampling')
    
    # Output arguments
    parser.add_argument('--save_trajectory', action='store_true',
                        help='Save all timesteps for visualization')
    
    args = parser.parse_args()
    
    # Run with error handling
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
            print('\nSuggestions:')
            print('1. Reduce --batch_size')
            print('2. Reduce --max_length')
            print('3. Enable --flash_mode for memory-efficient sampling')
            print('='*60 + '\n')
            sys.exit(1)
        else:
            raise e
