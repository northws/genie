# Flash IPA - Memory Efficient Invariant Point Attention
# Modified to support PyTorch 2.9
# Original source: flash_ipa package

from .ipa import InvariantPointAttention, IPAConfig, FA3_AVAILABLE, _is_hopper_gpu, _get_gpu_arch_name
from .edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
from .rigid import Rigid, create_rigid
from .factorizer import LinearFactorizer
from .linear import Linear
from .utils import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE

__all__ = [
    'InvariantPointAttention',
    'IPAConfig',
    'EdgeEmbedder',
    'EdgeEmbedderConfig',
    'Rigid',
    'create_rigid',
    'LinearFactorizer',
    'Linear',
    'ANG_TO_NM_SCALE',
    'NM_TO_ANG_SCALE',
    'FA3_AVAILABLE',
    '_is_hopper_gpu',
    '_get_gpu_arch_name',
]
