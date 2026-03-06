"""
Robust ground filtering for TLS landslide data.
- Combines geometry (local plane), intensity, and number_of_returns.
"""

import numpy as np
import laspy
from scipy.spatial import cKDTree

# -----------------------------
# User settings
# -----------------------------

input_file = "/Users/kerseynii/Library/CloudStorage/OneDrive-TUM/thesis/data_work/Thesis_Nortey/segmented_valley/segmented_valley_no_buildings.las"

RADIUS = 3.0               # neighbourhood radius (meters)
N_ITER = 3                 # robust iterations
SIGMA_R = 0.3              # controls residual weighting
GROUND_NUMRETURNS = [1] # 1 and 2 [1, 2] returns treated as ground-like, but in this case, 1 is chosen
RESIDUAL_THRESHOLD = 0.4   # classification threshold (meters)
MAX_POINTS = None          # or e.g. 150000 for testing

# -----------------------------
# Helper functions
# -----------------------------


def fit_plane_weighted(x, y, z, w):
    """Fit plane z = a*x + b*y + c (weighted least squares)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    w_sqrt = np.sqrt(w)
    Aw = A * w_sqrt[:, None]
    zw = z * w_sqrt
    params, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
    return params  # a, b, c


def residual_weight(r, sigma):
    """Points above the surface lose weight; others keep weight ~1."""
    w = np.ones_like(r, dtype=float)
    mask = r > 0
    w[mask] = np.exp(-(r[mask] / sigma) ** 2)
    return w


def intensity_weight(I, base_mask):
    """Weight based on local ground-like intensities."""
    if not np.any(base_mask):
        return np.ones_like(I, dtype=float)

    I_ground = I[base_mask]
    I_low = np.percentile(I_ground, 10)
    I_high = np.percentile(I_ground, 90)

    if I_high <= I_low:
        return np.ones_like(I, dtype=float)

    I_clipped = np.clip(I, I_low, I_high)
    t = (I_clipped - I_low) / (I_high - I_low)  # [0,1]
    return 0.2 + 0.8 * t  # map [0,1] → [0.2,1.0]


def echo_weight(num_ret):
    """1–2 returns get full weight, others get lower weight."""
    w = np.ones_like(num_ret, dtype=float)
    w[~np.isin(num_ret, GROUND_NUMRETURNS)] = 0.4
    return w


# -----------------------------
# Main
# -----------------------------


def main():
    print("Loading LAS file...")
    las = laspy.read(input_file)

    X = las.x
    Y = las.y
    Z = las.z
    I = np.array(las.intensity)
    NR = np.array(las.number_of_returns)

    N_total = len(Z)
    N = N_total if MAX_POINTS is None else min(MAX_POINTS, N_total)
    print(f"Using {N} points out of {N_total}.")

    X, Y, Z, I, NR = X[:N], Y[:N], Z[:N], I[:N], NR[:N]

    print("Building cKDTree on XY (fast)...")
    coords = np.vstack([X, Y]).T
    tree = cKDTree(coords)

    residuals = np.full(N, np.nan, dtype=float)

    print("Filtering points...")
    for i in range(N):
        # Find neighbours within RADIUS using KD-tree
        idx = tree.query_ball_point(coords[i], r=RADIUS)
        if len(idx) < 3:
            continue

        x_nb = X[idx]
        y_nb = Y[idx]
        z_nb = Z[idx]
        I_nb = I[idx]
        NR_nb = NR[idx]

        w = np.ones(len(idx), dtype=float)

        # Robust plane fitting
        for _ in range(N_ITER):
            a, b, c = fit_plane_weighted(x_nb, y_nb, z_nb, w)
            z_surf = a * x_nb + b * y_nb + c
            r = z_nb - z_surf

            w_r = residual_weight(r, SIGMA_R)
            base = (r <= 0) & np.isin(NR_nb, GROUND_NUMRETURNS)
            w_I = intensity_weight(I_nb, base)
            w_E = echo_weight(NR_nb)

            w = w_r * w_I * w_E

        # Final residual for point i
        z_surf_i = a * X[i] + b * Y[i] + c
        residuals[i] = Z[i] - z_surf_i

        if (i + 1) % 50000 == 0:
            print(f"Processed {i + 1}/{N}")

    print("Residual computation complete.")

    # Classification
    ground_mask = (residuals <= RESIDUAL_THRESHOLD) & ~np.isnan(residuals)
    veg_mask = ~ground_mask

    print(f"Ground points: {ground_mask.sum()}")
    print(f"Vegetation points: {veg_mask.sum()}")

    # Save ground
    ground_las = laspy.LasData(las.header)
    ground_las.points = las.points[:N][ground_mask]
    ground_las.write("ground_weighted_surface_kdtree_ground.las")

    # Save vegetation
    veg_las = laspy.LasData(las.header)
    veg_las.points = las.points[:N][veg_mask]
    veg_las.write("ground_weighted_surface_kdtree_vegetation.las")

    print("Done! Files saved.")


if __name__ == "__main__":
    main()
