import numpy as np
from skimage.feature import corner_fast
from .phasecong3 import phasecong3
from matching.detectors.rift.RIFT_descriptor_no_rotation_invariance import RIFT_descriptor_no_rotation_invariance

def detect(im1, im2, s, o, max_points=5000):
    """
    This is a simplest implementation of the proposed RIFT algorithm. In this implementation,
    rotation invariance part and corner point detection are not included.
    """

    # m1 and m2 are the maximum moment maps;
    # eo1[s,o] = convolution result for scale s and orientation o.
    # The real part is the result of convolving with the even symmetric filter,
    # the imaginary part is the result from convolution with the odd symmetric filter.

    m1, _, _, _, _, eo1, _, _ = phasecong3(im1, s, o, 3, mult=1.6, sigmaOnf=0.75, g=3, k=1)
    m2, _, _, _, _, eo2, _, _ = phasecong3(im2, s, o, 3, mult=1.6, sigmaOnf=0.75, g=3, k=1)

    # Normalize m1 and m2
    m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1))
    m2 = (m2 - np.min(m2)) / (np.max(m2) - np.min(m2))

    # FAST detector on the maximum moment maps to extract edge feature points.
    m1_heatmap: np.ndarray = corner_fast(m1, threshold=0.05)
    m2_heatmap: np.ndarray = corner_fast(m2, threshold=0.05)

    def select_strongest(map: np.ndarray, max_points_count):
        indices = np.argpartition(-map.ravel(),max_points)
        indices = np.unravel_index(indices, map.shape)
        indices = np.vstack(indices[::-1]).T
        return indices[:max_points_count, ...]



    # Select strongest points (assuming corner_fast returns coordinates in y, x order)
    m1_points = select_strongest(m1_heatmap, max_points) # number of keypoints can be set by users
    m2_points = select_strongest(m2_heatmap, max_points)
    return m1_points, eo1, m2_points, eo2


def describe(im1, m1_points, eo1, im2, m2_points, eo2,patch_size, s, o):
    # RIFT descriptor
    des_m1 = RIFT_descriptor_no_rotation_invariance(im1, m1_points, eo1, patch_size, s, o).astype(np.float32)
    des_m2 = RIFT_descriptor_no_rotation_invariance(im2, m2_points, eo2, patch_size, s, o).astype(np.float32)

    return des_m1, des_m2
