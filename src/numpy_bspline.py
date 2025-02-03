import numpy as np


def bspline_weights_vectorized(t_array: np.ndarray):
    """
    Vectorized version of bspline_weights(...) for t in [0..1].
    t_array: shape (N,)
    returns: shape (N,4), i.e. [w0, w1, w2, w3] per row
    """
    t_array = np.clip(t_array, 0.0, 1.0)
    t  = t_array
    t2 = t * t
    t3 = t2 * t

    w0 = (1 - 3*t + 3*t2 - t3) / 6.0
    w1 = (4 - 6*t2 + 3*t3) / 6.0
    w2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0
    w3 = t3 / 6.0

    return np.vstack([w0, w1, w2, w3]).T  # shape (N,4)

# ------------------------------------------------------------------
# Build a (4,4,4,3) corner-lookup array from the 8 corners_offset
# ------------------------------------------------------------------
def build_corner_table(corners_offset):
    """
    Build a (4,4,4,3) array for the offsets used by each (dU,dV,dW) in [-1..2].
    corner_table[dW,dV,dU] = corners_offset[(clamped_u, clamped_v, clamped_w)] 
    in R^3 (x,y,z).
    """
    corner_table = np.zeros((4, 4, 4, 3), dtype=float)
    for dW in range(-1, 3):  # -1..2
        for dV in range(-1, 3):
            for dU in range(-1, 3):
                # clamp to 0..1
                w_idx = 0 if dW < 0 else (1 if dW > 1 else dW)
                v_idx = 0 if dV < 0 else (1 if dV > 1 else dV)
                u_idx = 0 if dU < 0 else (1 if dU > 1 else dU)
                corner_table[dW+1, dV+1, dU+1] = corners_offset[(u_idx, v_idx, w_idx)]
    return corner_table

# ------------------------------------------------------------------
# Vectorized 2x2x2 B-spline deformation
# ------------------------------------------------------------------
def bspline_2x2x2_deform_fast(points, box_min, box_max, corners_offset):
    """
    Vectorized version of the 2x2x2 B-spline deform.
    points: (N,3) float
    corners_offset: dict((i,j,k), (3,)) i/j/k in {0,1}
    returns: (N,3)
    """
    # Build (4,4,4,3) corner lookup table once
    corner_table = build_corner_table(corners_offset)

    # Normalize points to [0,1] in each axis
    box_size = box_max - box_min
    normed = (points - box_min) / box_size
    normed = np.clip(normed, 0.0, 1.0)

    # Integral indices = floor(u), floor(v), floor(w), each in {0,1}
    ui = np.floor(normed[:, 0]).astype(int)
    vi = np.floor(normed[:, 1]).astype(int)
    wi = np.floor(normed[:, 2]).astype(int)

    # Fractional part
    frac_u = normed[:, 0] - ui
    frac_v = normed[:, 1] - vi
    frac_w = normed[:, 2] - wi

    # Compute B-spline weights for each axis
    Wu = bspline_weights_vectorized(frac_u)  # shape (N,4)
    Wv = bspline_weights_vectorized(frac_v)  # shape (N,4)
    Ww = bspline_weights_vectorized(frac_w)  # shape (N,4)

    # Accumulate deformation
    N = points.shape[0]
    delta = np.zeros((N, 3), dtype=float)

    # Only 64 combos (4x4x4), but each combo is applied to all N points at once
    for dW in range(4):
        for dV in range(4):
            for dU in range(4):
                # Weight shape (N,) after broadcast
                weight = Wu[:, dU] * Wv[:, dV] * Ww[:, dW]
                # corner_table[dW, dV, dU] is shape (3,)
                # multiply it across all N
                delta += weight[:, None] * corner_table[dW, dV, dU]

    # Add offset to original points
    deformed = points + delta
    return deformed
