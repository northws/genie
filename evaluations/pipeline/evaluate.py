import argparse
import os
import sys
import types
import torch

# HACK: Patch torch._six for deepspeed compatibility
if not hasattr(torch, '_six'):
    torch._six = types.ModuleType('torch._six')
    torch._six.inf = torch.inf
    sys.modules['torch._six'] = torch._six

# Add current directory to path so we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import Pipeline
from fold_models.esmfold import ESMFold
from inverse_fold_models.proteinmpnn import ProteinMPNN

def main(args):
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Initialize models
    print("Initializing ESMFold...")
    # ESMFold uses .cuda() in init, so it will pick up the visible device
    fold_model = ESMFold()

    print("Initializing ProteinMPNN...")
    # ProteinMPNN uses 'cuda:0' by default, which maps to the first visible device
    inverse_fold_model = ProteinMPNN(device='cuda:0')

    # Initialize pipeline
    pipeline = Pipeline(
        inverse_fold_model=inverse_fold_model,
        fold_model=fold_model
    )

    # Run evaluation
    pipeline.evaluate(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input samples')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('-g', '--gpus', type=str, default=None, help='GPU devices to use (e.g., "0,1")')
    parser.add_argument('-c', '--config', type=str, help='Config file (ignored but accepted for compatibility)')

    args = parser.parse_args()
    try:
        main(args)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('\n' + '='*60)
            print('CRITICAL ERROR: CUDA Out of Memory (OOM) during evaluation.')
            print('='*60)
            print('This pipeline uses ESMFold which is memory intensive.')
            print('Try freeing up GPU memory or running on a GPU with more VRAM.')
            print('='*60 + '\n')
            sys.exit(1)
        else:
            raise e
