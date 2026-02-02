from typing import Literal
from pathlib import Path
from .detectors.sift import SIFT
from .detectors.rift2.rift2 import RIFT2
from .detectors.dedode import DeDoDe
from .roma import Roma, CustomRoma


MatcherName = Literal['roma', 'sift', 'dedode', 'rift', 'roma_custom']

def make_matcher(matcher_name: MatcherName):
    if matcher_name == 'roma':
        return Roma(max_kpts=4096, coarse_res=112 * 4, upsample_res=112 * 4, use_center_crop=False)
    if matcher_name == 'roma_custom':
        return CustomRoma(Path('../checkpoints/roma.pth'), # alter if needed
                          coarse_res=224, upsample_res=224, use_center_crop=False)
    if matcher_name == 'dedode':
        return DeDoDe(num_features=4096, matcher='nn')
    if matcher_name == 'sift':
        return SIFT(True, 4096)
    if matcher_name == 'rift':
        return RIFT2(npt=4096, lowes_ratio=0.95)
    raise ValueError(f'Unknown matcher: {matcher_name}')
