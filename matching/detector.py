import torch
from typing import Callable, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectedPoints:
    points_a: np.ndarray
    points_b: np.ndarray
    matching_percentage: float
    aux_data: Any | None = None

    def __post_init__(self):
        assert len(self.points_a) == len(self.points_b)

    def avg_points_error(self):
        return np.linalg.norm(self.points_a - self.points_b, axis=1).mean()

    def get_inliers_mask(self, delta) -> np.ndarray[bool]:
        distances = np.linalg.norm(self.points_a - self.points_b, axis=1)
        # noinspection PyTypeChecker
        mask: np.ndarray[bool] = distances <= delta
        return mask

    def cmr(self, delta=5):
        mask = self.get_inliers_mask(delta=delta)
        if not mask.any():
            return 0.0
        return np.count_nonzero(mask) / len(mask)

    def localization_error(self, delta=5):
        mask = self.get_inliers_mask(delta)
        if not mask.any():
            return 0.0
        inliers_a = self.points_a[mask]
        inliers_b = self.points_b[mask]
        distances = np.linalg.norm(inliers_a - inliers_b, axis=1)
        return distances.mean()


Detector = Callable[[torch.Tensor, torch.Tensor], DetectedPoints | None]
