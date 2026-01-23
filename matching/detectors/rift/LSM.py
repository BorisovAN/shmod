import numpy as np
from scipy.linalg import qr, solve


def LSM(match1, match2, change_form):
    """
    Get the transformation parameters based on two sets of matches using the least square method.

    Parameters:
    match1 (numpy.ndarray): First set of matches
    match2 (numpy.ndarray): Second set of matches
    change_form (str): Type of transformation ('affine', 'perspective', or 'similarity')

    Returns:
    tuple: (parameters, rmse)
        parameters (numpy.ndarray): Transformation parameters
        rmse (float): Root Mean Square Error
    """
    match1_xy = match1[:, :2]
    match2_xy = match2[:, :2]

    A = np.repeat(match1_xy, 2, axis=0)
    B = np.tile([[1, 1, 0, 0], [0, 0, 1, 1]], (match1_xy.shape[0], 1))
    A = A * B
    B = np.tile([[1, 0], [0, 1]], (match1_xy.shape[0], 1))
    A = np.hstack((A, B))

    b = match2_xy.T.flatten()

    if change_form == 'affine':
        Q, R = qr(A)
        parameters = solve(R, Q.T @ b)
        parameters = np.append(parameters, [0, 0])

        N = match1.shape[0]
        match1_test = match1[:, :2]
        match2_test = match2[:, :2]
        M = parameters[:4].reshape(2, 2)
        match1_test_trans = M @ match1_test.T + parameters[4:6].reshape(2, 1)
        match1_test_trans = match1_test_trans.T
        test = match1_test_trans - match2_test
        rmse = np.sqrt(np.sum(test ** 2) / N)

    elif change_form == 'perspective':
        temp_1 = np.repeat(-match1_xy, 2, axis=0)
        temp_2 = np.tile(b, (1, 2))
        temp = temp_1 * temp_2
        A = np.hstack((A, temp))
        Q, R = qr(A)
        parameters = solve(R, Q.T @ b)

        N = match1.shape[0]
        match1_test = np.vstack((match1_xy.T, np.ones(N)))
        M = np.vstack((parameters[:5].reshape(2, 3),
                       [parameters[6], parameters[7], 1]))
        match1_test_trans = M @ match1_test
        match1_test_trans = match1_test_trans[:2] / match1_test_trans[2]
        match1_test_trans = match1_test_trans.T
        match2_test = match2_xy
        test = match1_test_trans - match2_test
        rmse = np.sqrt(np.sum(test ** 2) / N)

    elif change_form == 'similarity':
        A = np.zeros((2 * match1_xy.shape[0], 4))
        for i in range(match1_xy.shape[0]):
            A[2 * i] = [match1_xy[i, 0], match1_xy[i, 1], 1, 0]
            A[2 * i + 1] = [match1_xy[i, 1], -match1_xy[i, 0], 0, 1]

        Q, R = qr(A)
        parameters = solve(R, Q.T @ b)
        parameters = np.append(parameters, [0, 0, 0, 0])
        parameters[4:6] = parameters[2:4]
        parameters[2] = -parameters[1]
        parameters[3] = parameters[0]

        N = match1.shape[0]
        match1_test = match1[:, :2]
        match2_test = match2[:, :2]
        M = parameters[:4].reshape(2, 2)
        match1_test_trans = M @ match1_test.T + parameters[4:6].reshape(2, 1)
        match1_test_trans = match1_test_trans.T
        test = match1_test_trans - match2_test
        rmse = np.sqrt(np.sum(test ** 2) / N)

    return parameters, rmse