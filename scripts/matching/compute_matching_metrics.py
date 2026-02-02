import argparse
from multiprocessing.pool import ThreadPool
from pathlib import Path
import numpy as np
from timm.models.metaformer import poolformer_m36

from matching import DetectedPoints, compute_homography
from dataclasses import dataclass, field, asdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('KEYPOINTS_DIR', type=Path)
    parser.add_argument('OUTPUT_PATH', type=Path, optional=True)
    parser.add_argument('--success-ace-threshold', type=float, default=40, help='ACE upper limit for successful matching')
    parser.add_argument('--mma-distance-threshold', type=float, default=3, help='Upper distance for MMA metric')
    parser.add_argument('--image-size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--num-threads', type=int, default=8)
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

FAILED_MATCHING_STATS = MatchingStats(np.inf, 0, np.inf, 0, [0 for _ in LE_DISTANCE_THRESHOLDS], [0 for _ in LE_DISTANCE_THRESHOLDS])

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

    homograpty = compute_homography(detected_points.points_a, detected_points.points_b)
    if homograpty is None:
        return result

    ace = homograpty.ace(args.image_size)
    if ace <= args.success_ace_threshold:
        result.ace = ace
        result.MMA = homograpty.inliers_percent

    return result





def main():
    args = parse_args()

    def get_out_path():
        if args.OUTPUT_PATH is not None:
            return args.OUTPUT_PATH
        return args.KEYPOINTS_DIR.parent / "matching_stats.txt"

    assert args.KEYPOINTS_DIR.is_dir()
    out_path = get_out_path()
    assert not out_path.is_dir()

    out_path.mkdir(parents=True, exist_ok=True)
    metrics = {m:[] for m in asdict(FAILED_MATCHING_STATS).keys()}


    with ThreadPool(args.num_threads or 1) as pool:
        for stats in pool.imap_unordered(lambda f: process_file(f, args), args.KEYPOINTS_DIR.glob('*.npy')):
            stats_dict = asdict(stats)
            for m in metrics:
                metrics[m].append(stats_dict[m])

    metrics = {m: np.array(metrics[m]) for m in metrics}
    result = {}

    result['SR'] = np.count_nonzero(metrics['ACE'] <= args.success_ace_threshold)
    result['ACE'] = np.nanmean(metrics['ACE'])
    result['MMA'] = np.nanmean(metrics['MMA'])
    result['pairs_count']=np.mean(metrics['pairs_count'])
    result['point_error'] = np.nanmean(metrics['point_error'])
    le = np.mean(metrics['LE'], axis=0)
    cmr = np.mean(metrics['CMR'], axis=0)

    names = ['SR', 'ACE', 'MMA', 'pairs_count', 'point_error'] # fixed order of columns in CSV

    for d, le, cmr in zip(LE_DISTANCE_THRESHOLDS, le, cmr):
        result[f'LE{d}'].append(le)
        result[f'CMR{d}'].append(cmr)
        names.extend([f'LE{d}', f'CMR{d}'])

    with open('matching_stats.csv', 'w') as f:
        f.write(','.join(names) + '\r\n')
        values_row = ','.join(str(result[n] for n in names)) + '\r\n'
        f.write(values_row)






if __name__ == '__main__':
    main()