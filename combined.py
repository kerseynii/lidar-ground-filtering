# -------------------------------------------------
# Combined Number-of-Returns + Intensity Filtering
# -------------------------------------------------

import numpy as np
import laspy
import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1: Load LiDAR Data
# -------------------------------------------------
input_file = "/Users/kerseynii/Library/CloudStorage/OneDrive-TUM/thesis/data_work/Thesis_Nortey/segmented_valley/segmented_valley_no_buildings.las"

las = laspy.read(input_file)
print(f"Total points loaded: {las.header.point_count}")

# Extract attributes
num_returns = np.array(las.number_of_returns)
intensity   = np.array(las.intensity)

# -------------------------------------------------
# Step 2: Explore Distributions (NumReturns & Intensity)
# -------------------------------------------------

# Number of Returns distribution
unique_vals = np.unique(num_returns)
print(f"Unique 'Number of Returns' values: {sorted(unique_vals)}")

plt.figure(figsize=(8, 5))
plt.hist(num_returns, bins=range(1, int(num_returns.max()) + 2),
         align='left', edgecolor='black')
plt.title('Distribution of Number of Returns')
plt.xlabel('Number of Returns')
plt.ylabel('Number of Points')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Intensity distribution
plt.figure(figsize=(8, 5))
plt.hist(intensity, bins=100, edgecolor='black')
plt.title('Intensity Distribution of LiDAR Points')
plt.xlabel('Intensity')
plt.ylabel('Number of Points')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# -------------------------------------------------
# Step 3: Define Thresholds for Both Attributes
# -------------------------------------------------
# You can tune these based on the histograms above

# Number-of-returns rule:
#   Example: ground tends to be single-return (1)
single_return_values = [1, 2]   # adjust if needed

# Intensity rule:
#   Example: from your previous experiment: 37000–70000
min_intensity = 37000
max_intensity = 70000

print(f"Using single-return values: {single_return_values}")
print(f"Using intensity filter: {min_intensity}–{max_intensity}")

# -------------------------------------------------
# Step 4: Build Boolean Masks for Each Rule
# -------------------------------------------------
is_single_return     = np.isin(num_returns, single_return_values)
is_intensity_ground  = (intensity >= min_intensity) & (intensity <= max_intensity)

# Score: each satisfied rule = +1
#   - score = 2 → both conditions satisfied (high-confidence ground)
#   - score = 1 → one condition satisfied (ambiguous but ground-like)
#   - score = 0 → none satisfied (likely vegetation / non-ground)
score = is_single_return.astype(int) + is_intensity_ground.astype(int)

ground_strict_mask  = (score == 2)   # both conditions satisfied
ground_relaxed_mask = (score >= 1)   # at least one condition satisfied
nonground_mask      = (score == 0)   # neither condition satisfied

print(f"Strict ground points (score = 2): {ground_strict_mask.sum()}")
print(f"Relaxed ground points (score ≥ 1): {ground_relaxed_mask.sum()}")
print(f"Non-ground / vegetation candidates (score = 0): {nonground_mask.sum()}")

# -------------------------------------------------
# Step 5: Save Filtered LAS Files (Preserving All Attributes)
# -------------------------------------------------

# Strict ground
strict_output = "ground_strict_numret_intensity.las"
las_strict = laspy.LasData(las.header)
las_strict.points = las.points[ground_strict_mask]
las_strict.write(strict_output)
print(f"Strict ground points saved to {strict_output}")

# Relaxed ground
relaxed_output = "ground_relaxed_numret_intensity.las"
las_relaxed = laspy.LasData(las.header)
las_relaxed.points = las.points[ground_relaxed_mask]
las_relaxed.write(relaxed_output)
print(f"Relaxed ground points saved to {relaxed_output}")

# Vegetation / non-ground candidates
veg_output = "vegetation_candidates_numret_intensity.las"
las_veg = laspy.LasData(las.header)
las_veg.points = las.points[nonground_mask]
las_veg.write(veg_output)
print(f"Vegetation-candidate points saved to {veg_output}")

# -------------------------------------------------
# Step 6: 3D Visualization (subset for speed)
# -------------------------------------------------
try:
    from mpl_toolkits.mplot3d import Axes3D

    max_vis_points = 100_000  # limit for plotting

    def visualize_mask(mask, title, color_by_intensity=False):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            print(f"No points to visualize for: {title}")
            return

        if len(idx) > max_vis_points:
            idx = np.random.choice(idx, size=max_vis_points, replace=False)

        xs = las.x[idx]
        ys = las.y[idx]
        zs = las.z[idx]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        if color_by_intensity:
            intens_vis = intensity[idx]
            sc = ax.scatter(xs, ys, zs, c=intens_vis, cmap='terrain', s=1)
            plt.colorbar(sc, ax=ax, label='Intensity')
        else:
            ax.scatter(xs, ys, zs, s=1)

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    # Visualize different classes
    visualize_mask(ground_strict_mask,  "Strict Ground (score = 2)", color_by_intensity=True)
    visualize_mask(ground_relaxed_mask, "Relaxed Ground (score ≥ 1)", color_by_intensity=True)
    visualize_mask(nonground_mask,      "Vegetation / Non-ground (score = 0)", color_by_intensity=True)

except ImportError:
    print("mpl_toolkits.mplot3d not available. Skipping 3D visualization.")
