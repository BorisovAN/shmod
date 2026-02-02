# Copyright (c) 2018, Jiayuan Li
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#     * Neither the name of the Università degli studi dell'Aquila nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Reference: https://github.com/LJY-RS/RIFT-multimodal-image-matching

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

        im1_points = im1_points[:, ::-1].copy()
        im2_points = im2_points[:, ::-1].copy()
        matchedPoints1 = im1_points[indices[:, 0]]
        matchedPoints2 = im2_points[indices[:, 1]]

        p = (2*len(matchedPoints1))/(len(im1_points)+len(im2_points))

        return DetectedPoints(matchedPoints1.astype(float), matchedPoints2.astype(float), p)