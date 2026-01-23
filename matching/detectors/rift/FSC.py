import numpy as np
from typing import Tuple, List
from matching.detectors.rift.LSM import LSM


def FSC(cor1: np.ndarray, cor2: np.ndarray, change_form: str, error_t: float) -> Tuple[
    np.ndarray, float, np.ndarray, np.ndarray]:
    M, N = cor1.shape

    if change_form == 'similarity':
        n = 2
        max_iteration = M * (M - 1) // 2
    elif change_form == 'affine':
        n = 3
        max_iteration = M * (M - 1) * (M - 2) // (2 * 3)
    elif change_form == 'perspective':
        n = 4
        max_iteration = M * (M - 1) * (M - 2) // (2 * 3)
    else:
        raise ValueError("Invalid change_form")

    iterations = min(10000, max_iteration)

    most_consensus_number = 0
    cor1_new = np.zeros((M, N))
    cor2_new = np.zeros((M, N))

    # Assuming LSM function is defined elsewhere
    # from LSM import LSM

    for _ in range(iterations):
        while True:
            a = np.random.randint(0, M, n)
            cor11 = cor1[a, :2]
            cor22 = cor2[a, :2]

            if n == 2:
                if (a[0] != a[1] and
                        np.any(cor11[0] != cor11[1]) and
                        np.any(cor22[0] != cor22[1])):
                    break
            elif n == 3:
                if (len(set(a)) == 3 and
                        np.all(np.any(cor11[i] != cor11[j]) for i in range(3) for j in range(i + 1, 3)) and
                        np.all(np.any(cor22[i] != cor22[j]) for i in range(3) for j in range(i + 1, 3))):
                    break
            elif n == 4:
                if (len(set(a)) == 4 and
                        np.all(np.any(cor11[i] != cor11[j]) for i in range(4) for j in range(i + 1, 4)) and
                        np.all(np.any(cor22[i] != cor22[j]) for i in range(4) for j in range(i + 1, 4))):
                    break

        parameters, _ = LSM(cor11, cor22, change_form)
        solution = np.array([
            [parameters[0], parameters[1], parameters[4]],
            [parameters[2], parameters[3], parameters[5]],
            [parameters[6], parameters[7], 1]
        ])

        match1_xy = np.vstack((cor1[:, :2].T, np.ones(M)))

        if change_form == 'perspective':
            match1_test_trans = solution @ match1_xy
            match1_test_trans = match1_test_trans[:2] / match1_test_trans[2]
            match1_test_trans = match1_test_trans.T
            match2_test = cor2[:, :2]
            test = match1_test_trans - match2_test
            diff_match2_xy = np.sqrt(np.sum(test ** 2, axis=1))
        else:
            t_match1_xy = solution @ match1_xy
            match2_xy = np.vstack((cor2[:, :2].T, np.ones(M)))
            diff_match2_xy = np.sqrt(np.sum((t_match1_xy - match2_xy) ** 2, axis=0))

        index_in = np.where(diff_match2_xy < error_t)[0]
        consensus_num = len(index_in)

        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in]
            cor2_new = cor2[index_in]

    # Delete duplicate point pairs
    _, i1 = np.unique(cor1_new[:, :2], axis=0, return_index=True)
    cor1_new = cor1_new[np.sort(i1)]
    cor2_new = cor2_new[np.sort(i1)]

    _, i2 = np.unique(cor2_new[:, :2], axis=0, return_index=True)
    cor1_new = cor1_new[np.sort(i2)]
    cor2_new = cor2_new[np.sort(i2)]

    parameters, rmse = LSM(cor1_new[:, :2], cor2_new[:, :2], change_form)
    solution = np.array([
        [parameters[0], parameters[1], parameters[4]],
        [parameters[2], parameters[3], parameters[5]],
        [parameters[6], parameters[7], 1]
    ])

    return solution, rmse, cor1_new, cor2_new
