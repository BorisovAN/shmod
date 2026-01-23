import cv2
from dataclasses import dataclass
import numpy as np


@dataclass
class HomographyResult:
    homography_matrix: np.ndarray

    src_points: np.ndarray
    dst_points: np.ndarray
    inliers_percent: float

    def ace(self, image_shape: tuple[int, int]) -> float:
        h, w = image_shape
        points = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, 0],
            [w - 1, h - 1],
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed_points = cv2.perspectiveTransform(points, self.homography_matrix)
        diff = (points - transformed_points).squeeze(1)
        distance: np.ndarray = np.linalg.norm(diff, axis=1)
        return distance.mean()

    def inliers_error(self):
        return np.linalg.norm(self.src_points - self.dst_points, axis=1).mean()


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray, homography_method=cv2.RANSAC,
                       **homography_kwargs) -> HomographyResult | None:
    src_points = src_points.reshape(-1, 1, 2)  #[..., ::-1]
    dst_points = dst_points.reshape(-1, 1, 2)  #[..., ::-1]
    if len(src_points) < 4 or len(dst_points) < 4:
        return None

    homography_matrix, inliers_mask = cv2.findHomography(src_points, dst_points, method=homography_method,
                                                         **homography_kwargs)
    if homography_matrix is None:
        return None
    inliers_mask = inliers_mask[:, 0].astype(bool)
    return HomographyResult(homography_matrix, src_points.squeeze(1)[inliers_mask], dst_points.squeeze(1)[inliers_mask],
                            np.count_nonzero(inliers_mask) / len(inliers_mask))
