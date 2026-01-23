import numpy as np


def RIFT_descriptor_no_rotation_invariance(im, kps, eo, patch_size, s, o):
    KPS = kps.T  # keypoints
    yim, xim = im.shape[:2]

    CS = np.zeros((yim, xim, o))  # convolution sequence
    for j in range(o):
        for i in range(s):
            CS[:, :, j] += np.abs(eo[i][j])

    MIM = np.argmax(CS, axis=2)  # MIM maximum index map

    des = np.zeros((36 * o, KPS.shape[1]), dtype=np.float32)  # descriptor (size: 6x6xo)
    kps_to_ignore = np.zeros(KPS.shape[1], dtype=bool)

    for k in range(KPS.shape[1]):
        x = round(KPS[0, k])
        y = round(KPS[1, k])

        x1 = max(0, x - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        x2 = min(x + patch_size // 2, im.shape[1])
        y2 = min(y + patch_size // 2, im.shape[0])

        if y2 - y1 != patch_size or x2 - x1 != patch_size:
            kps_to_ignore[k] = True
            continue

        patch = MIM[y1:y2, x1:x2]  # local MIM patch for feature description
        ys, xs = patch.shape

        ns = 6
        RIFT_des = np.zeros((ns, ns, o), dtype=np.float32)  # descriptor vector

        # histogram vectors
        for j in range(ns):
            for i in range(ns):
                clip = patch[round(j * ys / ns):round((j + 1) * ys / ns),
                       round(i * xs / ns):round((i + 1) * xs / ns)]
                RIFT_des[j, i, :] = np.histogram(clip, bins=range(1, o + 2))[0]

        RIFT_des = RIFT_des.flatten()

        norm = np.linalg.norm(RIFT_des)
        if norm != 0:
            RIFT_des = RIFT_des / norm

        des[:, k] = RIFT_des

    # des = {
    #     'kps': KPS[:, ~kps_to_ignore].T,
    #     'des': des[:, ~kps_to_ignore].T
    # }

    #des[:, kps_to_ignore] = np.nan
    des = des[:, ~kps_to_ignore].copy()




    return des.T