import numpy as np
from scipy import spatial

# TODO: This should be placed in other place

def lineseg_dists(points:np.ndarray, line_points_1:np.ndarray, line_points_2:np.ndarray) -> np.ndarray:
    """Cartesian distance from point to line segment
    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Arguments:
        - points: np.array of shape (n_p, 2)
        - line_points_1: np.array of shape (n_l, 2)
        - line_points_2: np.array of shape (n_l, 2)

    Returns:
        - o: np.array of shape (n_p, n_l)
    """
    p, a, b = points, line_points_1, line_points_2
    if len(p.shape) < 2:
        p = p.reshape(1,2)
    n_p, n_l = p.shape[0], a.shape[0]
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    # signed parallel distance components, rowwise dot products of 2D vectors
    s = np.multiply(np.tile(a, (n_p,1)) - p.repeat(n_l, axis=0), np.tile(d, (n_p,1))).sum(axis=1)
    t = np.multiply(p.repeat(n_l, axis=0) - np.tile(b, (n_p,1)), np.tile(d, (n_p,1))).sum(axis=1)
    # clamped parallel distance
    h = np.amax([s, t, np.zeros(s.shape[0])], axis=0)
    # perpendicular distance component, rowwise cross products of 2D vectors  
    d_pa = p.repeat(n_l, axis=0) - np.tile(a, (n_p,1))
    c = d_pa[:, 0] * np.tile(d, (n_p,1))[:, 1] - d_pa[:, 1] * np.tile(d, (n_p,1))[:, 0]
    return np.hypot(h, c).reshape(n_p, n_l)

