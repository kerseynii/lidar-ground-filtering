# LiDAR Ground Filtering

This repository contains Python scripts for experimenting with **ground filtering in LiDAR point clouds**.  
The goal is to separate **terrain (ground) points** from **vegetation and other non-ground objects** in order to support the generation of **Digital Terrain Models (DTMs)**.

LiDAR data often contains reflections from vegetation, buildings, and other structures. To obtain a clean terrain surface, these non-ground points must be removed.  
The scripts in this project explore different filtering strategies that combine **geometric surface fitting** with additional LiDAR attributes such as **intensity** and **echo distribution**.

---

## Project Overview

The repository implements two main approaches to LiDAR ground filtering:

1. **Robust local surface fitting**
2. **Rule-based attribute filtering**

Both methods operate on LiDAR point cloud data stored in **LAS format**.

---

## Method 1: Robust Ground Filtering (`abground.py`)

This script estimates the terrain surface using **local plane fitting with iterative weighting**.

For each LiDAR point:

1. A neighborhood of nearby points is identified using a **KD-tree**.
2. A **local plane** is fitted to the neighborhood using weighted least squares.
3. The algorithm iteratively adjusts point weights based on:
   - **Residual distance** from the fitted surface
   - **Intensity values**
   - **Number of returns**
4. After several iterations, the fitted surface converges toward the local terrain.

Points are then classified as **ground or non-ground** based on their **final residual distance** from the estimated surface.

### Features used for weighting

**Geometry (Residuals)**  
Points above the estimated surface are more likely to represent vegetation and are down-weighted.

**Intensity Values**  
Ground surfaces often produce more consistent reflectance than vegetation. Intensity values help identify ground-like returns.

**Number of Returns**  
Single-return pulses are often associated with ground reflections, while multiple returns frequently occur in vegetation.

These features are combined to create a robust terrain estimation.

---

## Method 2: Rule-Based Filtering (`combined.py`)

The second script implements a simpler filtering approach based purely on LiDAR attributes.

Points are classified using rules based on:

- **Number of returns**
- **Intensity values**

For example:

- Points with **single returns** are treated as more likely ground.
- **Intensity thresholds** identify points that fall within a ground-like reflectance range.

This approach is computationally simpler and useful for quick experiments or parameter tuning.

---

## Workflow

Typical processing workflow:

1. Load LiDAR point cloud (`.las` file)
2. Extract attributes such as:
   - coordinates
   - intensity
   - number of returns
3. Apply a filtering algorithm
4. Separate points into:
   - **ground points**
   - **vegetation / non-ground points**
5. Export filtered point clouds

The filtered ground points can then be used to generate **Digital Terrain Models (DTMs)**.

---

## Algorithm Overview

### Robust Ground Filtering (`abground.py`)

```
LiDAR Point Cloud (.las)
        │
        ▼
Build spatial index (KD-tree)
        │
        ▼
Find neighbouring points within radius
        │
        ▼
Fit local plane using weighted least squares
        │
        ▼
Update weights using:
 - residual distance from surface
 - intensity values
 - number of returns
        │
        ▼
Iterate surface fitting
        │
        ▼
Compute residual for each point
        │
        ▼
Classify points
Ground (terrain)  /  Non-ground (vegetation)
```

---

### Rule-Based Filtering (`combined.py`)

```
LiDAR Point Cloud (.las)
        │
        ▼
Extract attributes
 - intensity
 - number of returns
        │
        ▼
Analyze distributions (histograms)
        │
        ▼
Apply rule-based thresholds
        │
        ▼
Assign score to each point
        │
        ▼
Classify points into:
 - strict ground
 - relaxed ground
 - vegetation candidates
```

## Requirements

The scripts require the following Python libraries:

- `numpy`
- `laspy`
- `scipy`

Install them with:

```bash
pip install numpy laspy scipy
```

---

## Running the Scripts

Example:

```bash
python thesis_abground.py
```

Make sure the input LAS file path is correctly specified in the script:

```python
input_file = "segmented_valley_no_buildings.las"
```

---

## Output

The scripts produce filtered point clouds containing:

- Ground points
- Vegetation / non-ground points

These outputs can be visualized in software such as:

- CloudCompare
- QGIS
- PDAL
- LAStools

---

## Possible Extensions

Potential improvements for this project include:

- Parallel processing for faster computation
- Adaptive neighborhood sizes
- Machine learning based classification
- Integration with PDAL pipelines
- Automatic Digital Terrain Model generation

---

## References

The implemented approaches are inspired by research on LiDAR ground filtering, including:

- **Kraus, K., & Pfeifer, N. (1998)**  
  *Determination of terrain models in wooded areas with airborne laser scanner data.*

- **Goepfert, J., Soergel, U., & Brzank, A. (2008)**  
  *Integration of intensity information and echo distribution in the filtering process of LiDAR data in vegetated areas.*

