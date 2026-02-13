from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
from multiprocessing.pool import ThreadPool

import numpy as np


from matching import DetectedPoints, compute_homography
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('KEYPOINTS_DIR', type=Path)
    parser.add_argument('--output-path', type=Path, required=False)
    parser.add_argument('--success-ace-threshold', type=float, default=40, help='ACE upper limit for successful matching')
    parser.add_argument('--mma-distance-threshold', type=float, default=3, help='Upper distance for MMA metric')
    parser.add_argument('--image-size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--threads', type=int, default=0)
    return parser.parse_args()

LE_DISTANCE_THRESHOLDS = [1, 2, 3, 4, 5]


@dataclass
class MatchingStats:
    point_error: float
    pairs_count: int
    ACE: float = np.nan
    MMA: float = np.nan
    LE: list[float] = field(default_factory=list)
    CMR: list[float] = field(default_factory=list)

FAILED_MATCHING_STATS = MatchingStats(np.nan, 0, np.nan, 0, [0 for _ in LE_DISTANCE_THRESHOLDS], [0 for _ in LE_DISTANCE_THRESHOLDS])

def process_file(f: Path, args):
    try:
        detected_points = np.load(f, allow_pickle=True).item()
        detected_points = DetectedPoints(**detected_points)
    except EOFError as e:
        # file exists but empty
        return FAILED_MATCHING_STATS

    result = MatchingStats(
        pairs_count=len(detected_points.points_a),
        point_error=detected_points.avg_points_error(),
        LE = [detected_points.localization_error(_) for _ in LE_DISTANCE_THRESHOLDS],
        CMR = [detected_points.cmr(_) for _ in LE_DISTANCE_THRESHOLDS]
    )

    homography = compute_homography(detected_points.points_a, detected_points.points_b)
    if homography is None:
        return result

    ace = homography.ace(args.image_size)
    if ace <= args.success_ace_threshold:
        result.ACE = ace
        result.MMA = homography.inliers_percent

    return result


def main():
    args = parse_args()

    def get_out_path():
        if args.output_path is not None:
            return args.output_path
        return args.KEYPOINTS_DIR.parent / "matching_stats.csv"

    assert args.KEYPOINTS_DIR.is_dir()
    out_path = get_out_path()
    if out_path.is_dir():
        out_path = out_path / "matching_stats.csv"




    metrics = {m:[] for m in asdict(FAILED_MATCHING_STATS).keys()}

    files = list(args.KEYPOINTS_DIR.glob('*.npy'))
    with ThreadPool(args.threads or 1) as pool:

        for stats in tqdm(pool.imap_unordered(lambda f: process_file(f, args), files), total=len(files)):
            stats_dict = asdict(stats)
            for m in metrics:
                metrics[m].append(stats_dict[m])

    metrics = {m: np.array(metrics[m]) for m in metrics}
    result = {}

    result['SR'] = np.count_nonzero(metrics['ACE'] <= args.success_ace_threshold)/len(files)
    result['ACE'] = np.nanmean(metrics['ACE'])
    result['MMA'] = np.nanmean(metrics['MMA'])
    result['pairs_count']=np.mean(metrics['pairs_count'])
    result['point_error'] = np.nanmean(metrics['point_error'])
    le = np.mean(metrics['LE'], axis=0)
    cmr = np.mean(metrics['CMR'], axis=0)

    names = ['SR', 'ACE', 'MMA', 'pairs_count', 'point_error'] # fixed order of columns in CSV

    for d, le, cmr in zip(LE_DISTANCE_THRESHOLDS, le, cmr):
        result[f'LE{d}'] = le
        result[f'CMR{d}'] = cmr

        names.extend([f'LE{d}', f'CMR{d}'])

    print("Writing results to ", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        names_row = ','.join(names) + '\r\n'
        values_row = ','.join(str(result[n]) for n in names) + '\r\n'
        print(names_row+ values_row)
        f.write(names_row)
        f.write(values_row)



if __name__ == '__main__':
    main()