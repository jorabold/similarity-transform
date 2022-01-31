import numpy as np
from typing import Tuple


def similarity_transform(
    X: np.ndarray, Y: np.ndarray, dim: int = 3
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Calculate the similarity transform between two (matching) point sets.

    Parameters
    ----------
    X: np.ndarray
        Points of first trajectory (dim x n matrix, where dim = 3 for 3D)
    Y: np.ndarray
        Points of second trajectory (dim x n matrix, where dim = 3 for 3D)
    dim: int
        Dimensionality of points

    Returns
    -------
    R: np.ndarray
        Rotation matrix from X to Y
    c: np.ndarray
        Scale factor from Y to Y
    t: np.ndarray
        Translation from X to Y

    Reference
    ---------
    S. Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns," in IEEE Transactions on Pattern Analysis
    and Machine Intelligence, vol. 13, no. 4, pp. 376-380, April 1991,
    doi: 10.1109/34.88573.
    """
    if X.shape[0] != dim:
        raise ValueError(
            f"You've set {dim=}, so X should have shape ({dim}xn) "
            + f"but is {X.shape}!"
        )
    if Y.shape[0] != dim:
        raise ValueError(
            f"You've set {dim=}, so Y should have shape ({dim}xn) but "
            + f"is {Y.shape}!"
        )
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have same shape! But {X.shape} != {Y.shape}"
        )
    m, n = X.shape

    mu_x = np.mean(X, axis=1)
    mu_y = np.mean(Y, axis=1)

    X_centered = (X.T - mu_x).T
    Y_centered = (Y.T - mu_y).T

    s_xx = np.mean(np.sum(X_centered**2, 0))
    Sigma_xy = 1 / n * X_centered @ Y_centered.T
    U, D, V = np.linalg.svd(Sigma_xy)
    V = V.T  # numpy has it the other way around as Umeyama
    D = np.diag(D)

    S = np.eye(m)
    if np.linalg.matrix_rank(Sigma_xy) > (m - 1):
        if np.linalg.det(Sigma_xy) < 0:
            S[m - 1, m - 1] = -1
    elif np.linalg.matrix_rank(Sigma_xy) == m - 1:
        if np.linalg.det(U) * np.linalg.det(V) < 0:
            S[m - 1, m - 1] = -1
    else:
        print("Rank too small! Cannot estimate transformation.")
        R = np.eye(m)
        c = 1
        t = np.zeros(m)
        return R, c, t

    R = (U @ S @ V.T).T
    c = np.trace(D @ S) / s_xx
    t = mu_y - c * R @ mu_x

    return R, c, t


def construct_transformation_matrix(
    R: np.ndarray, c: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Get transformation matrix from rotation R, scale c and translation."""
    n = R.shape[0]
    T = np.identity(n + 1)
    T[:n, :n] = c * R
    T[:n, n] = t
    return T


def apply_transformation(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform points with transformation matrix T.

    Parameters
    ----------
    points: np.ndarray
        3 x n matrix containing the points to transform
    T: np.ndarray
        4 x 4 transformation matrix in homogeneous coordinates

    Returns
    -------
    np.ndarray
        3 x n matrix containing the transformed points
    """
    m = points.shape[0]
    if m != 3:
        raise ValueError(f"points should be (3xn) but is {points.shape}!")
    src = np.ones((m + 1, points.shape[1]))
    src[:m, :] = np.copy(points)
    src = np.dot(T, src)
    return src[:m, :] / src[m:, :]
