import cv2
import torch
import numpy as np
from matching.detector import DetectedPoints

DEFAULT_MATCHER = cv2.BFMatcher(cv2.NORM_L2)


class SIFT:

    def __init__(self, use_cv_for_grayscale: bool = False, max_points: int = 256, matcher=DEFAULT_MATCHER,
                 lowes_ratio: float = 0.75, **sift_params):

        self.lowes_ratio = lowes_ratio
        self.use_cv_for_grayscale = use_cv_for_grayscale
        self.sift = cv2.SIFT.create(max_points, **sift_params)
        self.matcher = matcher

    def _preprocess(self, ref: torch.Tensor, tgt: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:

        ref = ref.detach().cpu().numpy()
        tgt = tgt.detach().cpu().numpy()

        # data = np.stack((ref, tgt), axis=0)
        # mean = np.mean(data)
        # std = np.std(data)
        # data_range = (mean-3*std, mean+3*std)
        # del data

        ref = np.moveaxis(ref, 0, -1)
        tgt = np.moveaxis(tgt, 0, -1)

        if ref.shape[-1] == 1:
            ref = ref[..., 0]
            tgt = tgt[..., 0]
        else:
            if self.use_cv_for_grayscale:
                ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
                tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
            else:
                ref = np.mean(ref, -1)
                tgt = np.mean(tgt, -1)
        # vmin = min(ref.min(), tgt.min())
        # vmax = max(ref.max(), tgt.max())
        # data_range = (vmin, vmax)
        # ref = np.clip(ref, *data_range)
        # tgt = np.clip(tgt, *data_range)
        # ref = (ref - data_range[0]) / (data_range[1] - data_range[0])
        # tgt = (tgt - data_range[0]) / (data_range[1] - data_range[0])

        return (ref * 255).astype(np.uint8), (tgt * 255).astype(np.uint8)

    def __call__(self, imA: torch.Tensor, imB: torch.Tensor) -> DetectedPoints | None:
        assert imA.shape == imB.shape
        assert imA.ndim in (3, 4)
        if imA.ndim == 4:
            assert imA.shape[0] == 1
        else:
            imA = imA[None]
            imB = imB[None]

        ref, tgt = self._preprocess(imA[0], imB[0])

        kpts1, des1 = self.sift.detectAndCompute(ref, None)
        kpts2, des2 = self.sift.detectAndCompute(tgt, None)

        if not kpts1 or not kpts2:
            return None

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good1 = []
        good2 = []
        for _ in matches:
            if len(_) == 1:
                m = _[0]
                good2.append(m.trainIdx)
                good1.append(m.queryIdx)
                continue
            m, n = _
            if m.distance <self.lowes_ratio* n.distance:
                good2.append(m.trainIdx)
                good1.append(m.queryIdx)

        if len(good1) == 0:
            return None

        kpt_out1 = []
        kpt_out2 = []

        for i, j in zip(good1, good2):
            kpt_out1.append(kpts1[i].pt[::-1])
            kpt_out2.append(kpts2[j].pt[::-1])

        matching_percentage = (len(set(good1)) + len(set(good2))) / (len(kpts1) + len(kpts2))
        return DetectedPoints(np.array(kpt_out1), np.array(kpt_out2), matching_percentage, (kpts1, kpts2))
