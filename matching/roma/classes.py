from pathlib import Path
from typing import Union

import numpy as np
import torch

from matching.detector import DetectedPoints
from matching.roma.models.matcher import RegressionMatcher
from matching.roma.models.model_zoo import roma_outdoor


class _RomaBase:
    def __init__(self, model: RegressionMatcher, target_size: int, max_kpts: int = 4096, normalize: bool = True,
                 use_center_crop: bool = True):

        self.use_center_crop = use_center_crop
        self.max_kpts = max_kpts
        self.normalize = normalize
        self.roma_model = model

        self.target_size = target_size

    @staticmethod
    def crop_to_size(img: torch.Tensor, target_size: int):
        B, C, H, W = img.shape

        if H > target_size:
            diff = H - target_size
            up = diff // 2
            bottom = diff - up
            img = img[:, :, up: -bottom, :]
        else:
            up = 0
        if W > target_size:
            diff = W - target_size
            left = diff // 2
            right = diff - left
            img = img[..., left:-right]
        else:
            left = 0

        assert img.shape[2:] == (target_size, target_size)

        return img, up, left

    def __call__(self, imA: torch.Tensor, imB: torch.Tensor):

        if self.use_center_crop:
            imA, up_A, left_A = self.crop_to_size(imA, self.target_size)
            imB, up_B, left_B = self.crop_to_size(imB, self.target_size)
        else:
            up_A, left_A, up_B, left_B = (0,) * 4

        imA = imA[0].moveaxis(0, -1).cpu().numpy()  # H, W, C
        imB = imB[0].moveaxis(0, -1).cpu().numpy()  # H, W, C

        warp, certainty = self.roma_model.match_arrays(imA, imB, device=torch.device('cuda'), normalize=self.normalize,
                                                       scale=1.0)

        keypoints = self.roma_model.sample(warp, certainty, self.max_kpts)[0]

        keypoints = self.roma_model.to_pixel_coordinates(keypoints, *imA.shape[:2], *imB.shape[:2])

        kpts_a, kpts_b = keypoints
        kpts_a = kpts_a.cpu().numpy()[:, ::-1]
        kpts_b = kpts_b.cpu().numpy()[:, ::-1]

        kpts_a += np.array([[up_A, left_A]])
        kpts_b += np.array([[up_B, left_B]])

        return DetectedPoints(kpts_a, kpts_b, 1.0)


class Roma(_RomaBase):

    def __init__(self, max_kpts: int = 4096, coarse_res: Union[int, tuple[int, int]] = 252,
                 upsample_res: Union[int, tuple[int, int]] = 388, normalize: bool = True,
                 amp_dtype: torch.dtype = torch.float16, use_center_crop: bool = True):

        roma_model = roma_outdoor(torch.device('cuda'), coarse_res=coarse_res, upsample_res=upsample_res,
                                  amp_dtype=amp_dtype).eval()
        super().__init__(model=roma_model, target_size=coarse_res, max_kpts=max_kpts, normalize=normalize,
                         use_center_crop=use_center_crop)

    @staticmethod
    def crop_to_size(img: torch.Tensor, target_size: int):
        B, C, H, W = img.shape

        if H > target_size:
            diff = H - target_size
            up = diff // 2
            bottom = diff - up
            img = img[:, :, up: -bottom, :]
        else:
            up = 0
        if W > target_size:
            diff = W - target_size
            left = diff // 2
            right = diff - left
            img = img[..., left:-right]
        else:
            left = 0

        assert img.shape[2:] == (target_size, target_size)

        return img, up, left

    def __call__(self, imA: torch.Tensor, imB: torch.Tensor):

        if self.use_center_crop:
            imA, up_A, left_A = self.crop_to_size(imA, self.target_size)
            imB, up_B, left_B = self.crop_to_size(imB, self.target_size)
        else:
            up_A, left_A, up_B, left_B = (0,) * 4

        imA = imA[0].moveaxis(0, -1).cpu().numpy()  # H, W, C
        imB = imB[0].moveaxis(0, -1).cpu().numpy()  # H, W, C

        warp, certainty = self.roma_model.match_arrays(imA, imB, device=torch.device('cuda'), normalize=self.normalize,
                                                       scale=1.0)

        keypoints = self.roma_model.sample(warp, certainty, self.max_kpts)[0]

        keypoints = self.roma_model.to_pixel_coordinates(keypoints, *imA.shape[:2], *imB.shape[:2])

        kpts_a, kpts_b = keypoints
        kpts_a = kpts_a.cpu().numpy()[:, ::-1]
        kpts_b = kpts_b.cpu().numpy()[:, ::-1]

        kpts_a += np.array([[up_A, left_A]])
        kpts_b += np.array([[up_B, left_B]])

        return DetectedPoints(kpts_a, kpts_b, 1.0)


class CustomRoma(_RomaBase):

    def __init__(self, weights_path: Path, max_kpts: int = 4096, coarse_res: Union[int, tuple[int, int]] = 224,
                 upsample_res: Union[int, tuple[int, int]] = 448, normalize: bool = False,
                 amp_dtype: torch.dtype = torch.bfloat16, use_center_crop: bool = True):
        weights = torch.load(weights_path)['model']
        roma_model = roma_outdoor(device=torch.device('cuda'), weights=weights, coarse_res=coarse_res,
                                  upsample_res=upsample_res, amp_dtype=amp_dtype).eval()
        super().__init__(model=roma_model, target_size=coarse_res, max_kpts=max_kpts, normalize=normalize,
                         use_center_crop=use_center_crop)
