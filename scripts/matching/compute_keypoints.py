import sys

import torch
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from matching import *
from models.type1 import Type1Model
from tqdm import tqdm
from data.image_folders_data_module import ImageFoldersDataModule

import argparse
MatcherName = Literal['roma', 'sift', 'dedode', 'rift', 'roma_custom']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matcher', choices=['roma', 'sift', 'dedode', 'rift', 'roma_custom'], default='sift')
    parser.add_argument('CKPT_PATH', type=Path)
    parser.add_argument('DATA_ROOT', type=Path)
    parser.add_argument('OUT_FOLDER', type=Path, optional=True)

    return parser.parse_args()


# CKPT_PATH = Path(sys.argv[1])
# DATA_ROOT = Path(sys.argv[2])
#
# MATCHER: MatcherName = 'sift'
# OUT_FOLDER: Path | None = Path(sys.argv[3]) if len(sys.argv) > 3 else None


def get_out_folder(args):
    if args.OUT_FOLDER is not None:
        return args.OUT_FOLDER
    return args.CKPT_PATH.parent / args.matcher / 'keypoints'


def save_keypoints(f: Path, points: DetectedPoints):
    out_dict = {
        'points_a': points.points_a,
        'points_b': points.points_b,
        'matching_percentage': points.matching_percentage
    }

    np.save(f, out_dict)


def main():
    args = parse_args()

    datamodule = ImageFoldersDataModule(args.DATA_ROOT, ['s1', 's2'], 1, 1)

    try:
        model = Type1Model.load_from_checkpoint(args.CKPT_PATH, strict=False)
    except:
        model = torch.load(args.CKPT_PATH, weights_only=False).cuda()

    loader = datamodule.test_dataloader()
    matcher = make_matcher(args.matcher)

    out_folder = get_out_folder(args)

    # noinspection PyTypeChecker
    pbar = tqdm(total=len(loader.dataset))
    out_folder.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = [b.cuda() for b in batch]
            s1_features, s2_features = model(batch)
            keypoints = matcher(s1_features, s2_features)
            if keypoints is not None:
                save_keypoints(out_folder / f'{i}.npy', keypoints)
            else:
                (out_folder / f'{i}.npy').touch() #
            pbar.update(1)
