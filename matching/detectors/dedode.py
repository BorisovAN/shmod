from typing import Literal

import kornia.feature
import torch

from matching.detector import Detector, DetectedPoints
from kornia.feature import DeDoDe as kDeDoDe


class DeDoDe(Detector):

    def __init__(self, matcher: Literal['nn', 'lightglue'] = 'nn', num_features=4096, edge_margin=5):
        self.edge_margin = edge_margin
        self.num_features = num_features
        self.dedode = kDeDoDe(descriptor_model='G').from_pretrained().cuda().eval()
        self.matching_method = matcher
        self.lg_mathcer = kornia.feature.LightGlueMatcher('dedodeg').eval().cuda() if matcher == 'lightglue' else None

    @staticmethod
    def normalize_image(x):
        #x = x - x.mean()
        x = x - 0.5
        x *= 2
        #x = x / 0.45#(x.std()+1e-7)
        return x

    @staticmethod
    def filter_keypoints(kpts: torch.Tensor, descriptors: torch.Tensor, img: torch.Tensor, margin=5):
        h, w = img.shape[-2:]
        right = w - 5
        top = h - 5
        h_mask = torch.logical_or(kpts[..., 0] < margin, kpts[..., 0] >= top)
        w_mask = torch.logical_or(kpts[..., 1] < margin, kpts[..., 1] >= right)

        mask = torch.logical_or(h_mask, w_mask)
        mask = torch.logical_not(mask)[0]
        indices = torch.nonzero(mask)[..., 0]

        return kpts[:, indices, :], descriptors[:, indices, :]

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor):
        #img1 = self.normalize_image(img1)
        #img2 = self.normalize_image(img2)
        keypoints1, _, descriptors1 = self.dedode(img1, self.num_features, apply_imagenet_normalization=True)
        keypoints2, _, descriptors2 = self.dedode(img2, self.num_features, apply_imagenet_normalization=True)
        # 1, 2024, 2
        if self.edge_margin > 0:
            keypoints1, descriptors1 = self.filter_keypoints(keypoints1, descriptors1, img1, self.edge_margin)
            keypoints2, descriptors2 = self.filter_keypoints(keypoints2, descriptors2, img2, self.edge_margin)

        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return None
        descriptors1, descriptors2 = descriptors1[0], descriptors2[0]
        if self.matching_method == 'nn':
            dists, idxs = kornia.feature.match_mnn(descriptors1, descriptors2)
        else:
            hw1 = torch.tensor(img1.shape[2:]).cuda()
            hw2 = torch.tensor(img2.shape[2:]).cuda()
            laf1 = kornia.feature.laf_from_center_scale_ori(keypoints1)
            laf2 = kornia.feature.laf_from_center_scale_ori(keypoints2)
            dists, idxs = self.lg_mathcer(descriptors1, descriptors2, laf1, laf2, hw1, hw2)
        if len(idxs) == 0:
            return None
        keypoints1, keypoints2 = keypoints1[0], keypoints2[0]
        matching_percentage = 2 * len(idxs) / (len(descriptors1) + len(keypoints2))
        keypoints1 = keypoints1[idxs[:, 0]].cpu().numpy()[:, ::-1]
        keypoints2 = keypoints2[idxs[:, 1]].cpu().numpy()[:, ::-1]

        return DetectedPoints(keypoints1, keypoints2, matching_percentage)
