import numpy as np
import torch
import cv2

from matching.detector import Detector,DetectedPoints
from matching.detectors.rift.RIFT_no_rotation_invariance import detect, describe

class RIFT(Detector):

    def __init__(self, max_points=4096, patch_size=64, s=4, o=6, neighbours_count=2, neighbour_distance_ratio=5):
        self.neighbour_distance_ratio = neighbour_distance_ratio
        self.neighbours_count =  neighbours_count
        self.o = o
        self.s = s
        self.patch_size = patch_size
        self.max_points = max_points
        self.matcher =cv2.BFMatcher(crossCheck=True)

    #
    # def detect(self, im1: np.ndarray, im2: np.ndarray):
    #     m1_points, eo1, m2_points, eo2 = detect(im1, im2, self.s, self.o,
    #                                             max_points=self.max_points)
    #     return m1_points, eo1, m2_points, eo2
    #
    # def describe(self):

    def __call__(self, im1: torch.Tensor, im2:torch.Tensor):

        im1: np.ndarray = im1.cpu().detach_().moveaxis(1, -1).numpy()[0]
        im2: np.ndarray = im2.cpu().detach_().moveaxis(1, -1).numpy()[0]

        im1 = (im1 * 255).astype(np.uint8)
        im2 = (im2 * 255).astype(np.uint8)

        im1_points, eo1, im2_points, eo2 = detect(im1, im2, self.s, self.o,
                                                     max_points=self.max_points)

        des1, des2 = describe(im1, im1_points, eo1, im2, im2_points, eo2, self.patch_size,
                              self.s, self.o)

        matches = self.matcher.knnMatch(des1, des2, k=self.neighbours_count)
        # Apply ratio test

        matches = [m for m in matches if len(m) > 0]
        if self.neighbours_count > 1:
            good_matches = []
            for m, n in matches:
                if m.distance < self.neighbour_distance_ratio * n.distance:
                    good_matches.append(m)

        else:
            good_matches = [m[0] for m in matches]
        indices = np.array([(m.queryIdx, m.trainIdx) for m in good_matches])

        if len(indices) == 0:
            indices = np.empty((0,2), dtype=int)
        _, i = np.unique(indices[:, 0], return_index=True)
        indices = indices[i, :]
        _, i = np.unique(indices[:, 1], return_index=True)
        indices = indices[i, :]

        # matched_indices1 = set(indices[:, 0])
        # matched_indices2 = set(indices[:, 1])

        # unmatched_mask1 = np.array( [(i not in matched_indices1) for i in range(len(pts.coords1))])
        # unmatched_mask2 = np.array([(i not in matched_indices2) for i in range(len(pts.coords2))])
        im1_points = im1_points[:, ::-1].copy()
        im2_points = im2_points[:, ::-1].copy()
        matchedPoints1 = im1_points[indices[:, 0]]
        matchedPoints2 = im2_points[indices[:, 1]]

        p = (2*len(matchedPoints1))/(len(im1_points)+len(im2_points))

        return DetectedPoints(matchedPoints1.astype(float), matchedPoints2.astype(float), p)