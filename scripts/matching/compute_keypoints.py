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

MatcherName = Literal['roma', 'sift', 'dedode', 'rift', 'roma_custom']
CKPT_PATH = Path(sys.argv[1])
DATA_ROOT = Path(sys.argv[2])

MATCHER: MatcherName = 'sift'
OUT_FOLDER: Path | None = Path(sys.argv[3]) if len(sys.argv) > 3 else None


def get_out_folder():
    if OUT_FOLDER is not None:
        return OUT_FOLDER
    return CKPT_PATH.parent / 'keypoints' / CKPT_PATH.stem


def save_keypoints(f: Path, points: DetectedPoints):
    out_dict = {
        'points_a': points.points_a,
        'points_b': points.points_b,
        'matching_percentage': points.matching_percentage
    }

    np.save(f, out_dict)


def main():
    datamodule = ImageFoldersDataModule(DATA_ROOT, ['s1', 's2'], 1, 1)

    try:
        model = Type1Model.load_from_checkpoint(CKPT_PATH, strict=False)
    except:
        model = torch.load(CKPT_PATH, weights_only=False).cuda()

    loader = datamodule.test_dataloader()
    matcher = make_matcher(MATCHER)

    out_folder = get_out_folder()

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
            pbar.update(1)
