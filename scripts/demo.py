import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent/'..'))

import cv2
import numpy as np
import torch

from data.image_folder import ImageFolder
from matching import MatcherName, make_matcher
from models.identity import MSIdentity, RGBIdentity
from models.log import Log10
from models.percentile_limiter import PercentileLimiter
from models.type1 import Type1Model

torch.set_float32_matmul_precision('high')
import argparse
from typing import get_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('DATA_ROOT', type=Path)
    parser.add_argument('CKPT_PATH', type=Path)

    parser.add_argument('--limit-net-output', action='store_true', help='Limit the output of net (required to mitigate the low dynamic range for the type-2 network)')
    parser.add_argument('--matcher', choices=list(get_args(MatcherName)),required=False)
    parser.add_argument('--max-kpts', type=int, default=256)
    return parser.parse_args()

limiter = PercentileLimiter(0.5, 99.5).cuda()
log  = Log10().cuda()

def convert_to_rgb(img: torch.Tensor, *, limit_range: bool = False, use_log_scale: bool = False):
    if use_log_scale:
        img = log(img)

    if limit_range:
        img = limiter(img)

    img = img[0]
    img = torch.clamp(img, 0, 1) * 255
    img = img.to(torch.uint8)
    img = torch.moveaxis(img, 0, -1)  # CHW -> HWC
    img = img.cpu().detach().numpy()
    return img


# noinspection PyDefaultArgument
def show_image(img: np.ndarray, window_name: str, *, _created_windows=set()):
    if window_name not in _created_windows:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        _created_windows.add(window_name)
    cv2.imshow(window_name, img)


def array_to_keypoints(arr: np.ndarray):
    return [
        cv2.KeyPoint(p[1], p[0],1)
        for p in arr
    ]

def draw_keypoints(img1, img2, kpts, mask, color, out_img):
    if not mask.any():
        return out_img
    pts_a = kpts.points_a[mask]
    pts_b = kpts.points_b[mask]
    return cv2.drawMatches(img1, array_to_keypoints(pts_a), img2, array_to_keypoints(pts_b),
                           [cv2.DMatch(i, i, 0) for i in range(len(pts_a))], out_img, color, [0, 0, 0], '',
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG if out_img is not None else 0
                           )

def get_image_with_matches(matcher, img1: torch.Tensor, img2: torch.Tensor, img1_rgb: np.ndarray, img2_rgb: np.ndarray):
    # find the keypoints and descriptors with SIFT
    kpts = matcher(img1, img2)

    if kpts is None:
        return np.concatenate((img1_rgb, img2_rgb), axis=1)

    inliers_mask = kpts.get_inliers_mask(5)
    outliers_mask = np.logical_not(inliers_mask)

    out_img = None

    if np.count_nonzero(inliers_mask)*0.5 <= np.count_nonzero(outliers_mask): # more rare cases shall not be hidden
        out_img = draw_keypoints(img1_rgb, img2_rgb, kpts, outliers_mask, [0,0,192], out_img)
        out_img = draw_keypoints(img1_rgb, img2_rgb, kpts, inliers_mask, [0,255,0], out_img)
    else:
        out_img = draw_keypoints(img1_rgb, img2_rgb, kpts, inliers_mask, [0, 255, ], out_img)
        out_img = draw_keypoints(img1_rgb, img2_rgb, kpts, outliers_mask, [0, 0, 192], out_img)
    assert out_img is not None
    return out_img



def main():
    args = parse_args()
    if str(args.CKPT_PATH) == "IDENTITY":
        model = MSIdentity()
    elif str(args.CKPT_PATH) == "IDENTITY_RGB":
        model = RGBIdentity()
    else:
        try:
            model = Type1Model.load_from_checkpoint(args.CKPT_PATH, strict=False)
        except Exception as E:
            model = torch.load(args.CKPT_PATH, weights_only=False).cuda()

    matcher = None
    if args.matcher is not None:
        assert args.max_kpts > 0
        matcher = make_matcher(args.matcher,args.max_kpts)

    model.eval()

    dataset = ImageFolder(args.DATA_ROOT, ['s1', 's2'])

    idx = 0
    with torch.no_grad():
        while True:
            data = tuple(_[None].cuda() for _ in dataset[idx])

            results = model(data)

            if args.limit_net_output:
                results = tuple(limiter(r) for r in results)
            s1, s2 = data
            r1, r2 = results

            s1 = convert_to_rgb(s1[:, (0, 1, 0), ...], limit_range=True, use_log_scale=True)
            s1[..., 0] = 0
            s2 = convert_to_rgb(s2[:, :3, ...], limit_range=True)
            r1 = convert_to_rgb(r1)
            r2 = convert_to_rgb(r2)

            if matcher is not None:
                image_with_matches = get_image_with_matches(matcher, *results, r1, r2)
                show_image(image_with_matches, 'matching result')

            show_image(s1, 's1')
            show_image(s2, 's2')
            show_image(r1, 'r1')
            show_image(r2, 'r2')

            while True:
                c = cv2.waitKey(0)
                if c == ord('q'):
                    exit(0)
                if c == ord('d'):
                    idx = min(idx + 1, len(dataset) - 1)
                    break
                if c == ord('a'):
                    idx = max(idx - 1, 0)
                    break

def main2():
    args = parse_args()
    if str(args.CKPT_PATH) == "IDENTITY":
        model = MSIdentity()
    elif str(args.CKPT_PATH) == "IDENTITY_RGB":
        model = RGBIdentity()
    else:
        try:
            model = Type1Model.load_from_checkpoint(args.CKPT_PATH, strict=False)
        except Exception as E:
            model = torch.load(args.CKPT_PATH, weights_only=False).cuda()

    # matcher = None
    # if args.matcher is not None:
    #     assert args.max_kpts > 0
    #     matcher = make_matcher(args.matcher, args.max_kpts)

    matcher_names = ('sift', 'dedode', 'roma', 'roma_custom', 'rift')
    matchers = [(m,make_matcher(m, 4096)) for m in matcher_names]

    model.eval()

    dataset = ImageFolder(args.DATA_ROOT, ['s1', 's2'])

    assert dataset is not None

    INDICES = [14386, 13374, 13373, 12360,12338, 12336, 101, 1218, 1275, 2317, 2321, 5333, 13339]
    OUT_ROOT = Path(args.CKPT_PATH.stem)
    with torch.no_grad():
        for idx in INDICES:
            out_folder = OUT_ROOT / str(idx)
            out_folder.mkdir(parents=True, exist_ok=True)
            data = tuple(_[None].cuda() for _ in dataset[idx])

            results = model(data)

            if args.limit_net_output:
                results = tuple(limiter(r) for r in results)
            s1, s2 = data
            r1, r2 = results
            s1 = log(s1)
            s1 = limiter(s1)

            s1 = convert_to_rgb(s1[:, (0, 1, 0), ...]) if str(args.CKPT_PATH) != 'IDENTITY_RGB' else convert_to_rgb(s1)
            s1[..., 0] = 0
            s2 = convert_to_rgb(s2[:, :3, ...], limit_range=True) if str(args.CKPT_PATH) != 'IDENTITY_RGB' else convert_to_rgb(s2)
            r1 = convert_to_rgb(r1)
            r2 = convert_to_rgb(r2)

            cv2.imwrite(str(out_folder / 's1.png'), s1)
            cv2.imwrite(str(out_folder / 's2.png'), s2)
            cv2.imwrite(str(out_folder / 'r1.png'), r1)
            cv2.imwrite(str(out_folder / 'r2.png'), r2)


            for matcher_name, matcher in matchers:
                image_with_matches = get_image_with_matches(matcher, *results, r1, r2)
                cv2.imwrite(str(out_folder / (matcher_name+'.png')), image_with_matches)


            # if matcher is not None:
            #     image_with_matches = get_image_with_matches(matcher, *results, r1, r2)
            #     show_image(image_with_matches, 'matching result')



if __name__ == "__main__":
    main2()
