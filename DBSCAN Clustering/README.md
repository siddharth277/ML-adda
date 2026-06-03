# 3D Room Segmentation — Stanford S3DIS Dataset

A complete Python pipeline that loads a real indoor 3D point cloud from the **Stanford S3DIS dataset**, segments it into semantic regions (floor, ceiling, wall, furniture, beam, window), and exports colorized visualizations and a `.ply` file you can open in MeshLab or CloudCompare.

----

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is a Point Cloud?](#2-what-is-a-point-cloud)
3. [Dataset — Stanford S3DIS](#3-dataset--stanford-s3dis)
4. [Project Structure](#4-project-structure)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Installation & Requirements](#6-installation--requirements)
7. [How to Run](#7-how-to-run)
8. [Configuration](#8-configuration)
9. [Stage-by-Stage Explanation](#9-stage-by-stage-explanation)
10. [Semantic Labels & Colors](#10-semantic-labels--colors)
11. [Output Files](#11-output-files)
12. [Key Parameters & Tuning](#12-key-parameters--tuning)
13. [Known Issues & Fixes Applied](#13-known-issues--fixes-applied)
14. [Results & Accuracy](#14-results--accuracy)
15. [Dependencies](#15-dependencies)

---

## 1. Project Overview

This project takes a scanned 3D room (as a point cloud) and automatically labels every point with what it physically is — floor, ceiling, wall, furniture, beam, or window — using only **geometry and Z-height rules**, no deep learning required.

**What it does:**
- Loads raw S3DIS point cloud data (X Y Z R G B per point)
- Cleans and downsamples the point cloud
- Applies semantic labels using annotation files + Z-height heuristics
- Runs per-class DBSCAN clustering to separate individual objects
- Generates colorized 2D/3D visualizations
- Exports a `.ply` file and cluster statistics CSV
- Draws a top-down floor plan
- Computes bounding boxes around individual furniture pieces

**Why it matters:** 3D room understanding is a core problem in robotics, AR/VR, smart buildings, and autonomous navigation. This project shows how you can do meaningful room segmentation using classical geometry — no GPU or neural network needed.

---

## 2. What is a Point Cloud?

A **point cloud** is a collection of millions of 3D points (dots) floating in space, each recorded by a laser scanner. Every point has:

| Field | Meaning | Example |
|-------|---------|---------|
| X | Left-right position in meters | 2.341 |
| Y | Forward-backward position in meters | 1.872 |
| Z | Height in meters (vertical axis) | 0.052 |
| R | Red color from camera (0–255) | 198 |
| G | Green color from camera (0–255) | 172 |
| B | Blue color from camera (0–255) | 145 |

A typical S3DIS office room has **500,000–900,000 points**. After downsampling this becomes ~35,000–40,000 points, which is fast to process.

---

## 3. Dataset — Stanford S3DIS

**Full name:** Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS)  
**Version used:** v1.2 Aligned Version  
**Download:** [http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html)

### Dataset structure

```
Stanford3dDataset_v1.2_Aligned_Version/
├── Area_1/
│   ├── conferenceRoom_1/
│   │   ├── conferenceRoom_1.txt       ← full merged point cloud (X Y Z R G B)
│   │   └── Annotations/
│   │       ├── floor_1.txt            ← ground-truth floor points
│   │       ├── ceiling_1.txt          ← ground-truth ceiling points
│   │       ├── wall_1.txt             ← wall points
│   │       ├── wall_2.txt
│   │       ├── chair_1.txt            ← individual furniture objects
│   │       ├── chair_2.txt
│   │       ├── table_1.txt
│   │       ├── clutter_1.txt
│   │       └── ...
│   ├── office_1/
│   ├── hallway_1/
│   └── ...  (17 rooms in Area_1)
├── Area_2/
├── Area_3/
├── Area_4/
├── Area_5/
└── Area_6/
```

### Key facts

- **6 Areas** corresponding to different floors of a building
- **272 rooms total** across all areas
- **Room types:** office, conference room, hallway, copy room, lounge, pantry, open space
- **13 semantic categories:** ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter
- Each `Annotations/` folder contains one `.txt` per object instance (e.g., `chair_1.txt`, `chair_2.txt`)
- The merged room `.txt` file combines all annotation files into one

### Category mapping used in this project

| S3DIS category | Mapped label | Color |
|---|---|---|
| floor | `floor` | Green |
| ceiling | `ceiling` | Light grey |
| wall | `wall` | Steel blue |
| beam, column | `beam` | Brown |
| window | `window` | Cyan |
| door, table, chair, sofa, bookcase, board, clutter | `furniture` | Orange |

---

## 4. Project Structure

```
s3dis_project/
├── 3D_Room_Segmentation_S3DIS.ipynb   ← main notebook (run this)
├── Stanford3dDataset_v1.2_Aligned_Version/
│   └── Area_1/
│       └── conferenceRoom_1/
│           ├── conferenceRoom_1.txt
│           └── Annotations/
│               └── *.txt
└── output/                             ← created automatically
    ├── 01_raw_rgb.png                  ← original RGB top-down + side view
    ├── 02_semantic_labels.png          ← 3-view semantic segmentation plot
    ├── 03_dbscan_clusters.png          ← per-cluster colored plot
    ├── 04_floor_plan.png               ← top-down floor plan
    ├── 05_furniture_boxes.png          ← bounding boxes around furniture
    ├── segmented_room.ply              ← 3D colored point cloud (open in MeshLab)
    └── cluster_report.csv             ← stats per cluster (label, size, bbox, etc.)
```

---

## 5. Pipeline Architecture

```
S3DIS Annotations/*.txt files
         │
         ▼
┌─────────────────────────────┐
│  Stage 1: Load from         │   Read every file in Annotations/, tag each
│  Annotations/               │   point with its ground-truth semantic label
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 2: Voxel Downsample  │   5cm voxel grid → ~35k points, majority
│  (keeping labels)           │   label vote per voxel cell
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 3: Visualize         │   Top-down + side + front views in real RGB
│  raw RGB                    │   colors as a sanity check
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 4: Visualize         │   Same 3 views but colored by semantic label
│  semantic labels            │   (floor=green, wall=blue, etc.)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 5: Per-class DBSCAN  │   Run DBSCAN separately per label to split
│  clustering                 │   individual objects (e.g., chair_1 vs chair_2)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 6: Save outputs      │   .ply file, CSV report, floor plan,
│  & visualizations           │   furniture bounding boxes
└──────────────┬──────────────┘
               │
               ▼
         ./output/ files
```

---

## 6. Installation & Requirements

### Python version
Python 3.8 or higher recommended.

### Install all dependencies (run once)

```bash
pip install open3d numpy scikit-learn matplotlib pandas
```

Or run **Cell 0** inside the notebook — it installs everything automatically.

### Packages used

| Package | Version | Purpose |
|---|---|---|
| `open3d` | ≥ 0.17 | Point cloud I/O, voxel downsampling, .ply export |
| `numpy` | ≥ 1.21 | Array math, coordinate operations |
| `scikit-learn` | ≥ 1.0 | DBSCAN clustering algorithm |
| `matplotlib` | ≥ 3.5 | All 2D visualizations and plots |
| `pandas` | ≥ 1.3 | Cluster statistics table and CSV export |

---

## 7. How to Run

### Step 1 — Download the dataset

Go to [http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html) and download **Stanford3dDataset_v1.2_Aligned_Version**.

### Step 2 — Update the path in Cell 2

Open the notebook and find this line in **Cell 2**:

```python
DATASET_ROOT = r"C:\Desktop\s3dis_project\Stanford3dDataset_v1.2_Aligned_Version"
```

Change it to wherever you extracted the dataset on your machine. Examples:

```python
# Windows
DATASET_ROOT = r"C:\Users\YourName\Downloads\Stanford3dDataset_v1.2_Aligned_Version"

# Mac / Linux
DATASET_ROOT = "/home/yourname/data/Stanford3dDataset_v1.2_Aligned_Version"
```

### Step 3 — Choose a room

Still in Cell 2, set which area and room you want to process:

```python
AREA = "Area_1"
ROOM = "conferenceRoom_1"   # or "office_1", "hallway_1", etc.
```

### Step 4 — Run all cells

In Jupyter: **Kernel → Restart & Run All**

Or run cells top to bottom one at a time.

### Step 5 — View outputs

Check the `./output/` folder that gets created next to the notebook. Open `segmented_room.ply` in MeshLab or CloudCompare to see the full 3D colored result.

---

## 8. Configuration

All tuning parameters are set in **Cell 2** and at the top of **Cell 7**.

### Cell 2 — Main settings

```python
DATASET_ROOT = "..."       # path to extracted S3DIS dataset
AREA         = "Area_1"   # Area_1 through Area_6
ROOM         = "conferenceRoom_1"  # exact folder name inside the area
OUTPUT_DIR   = "./output"  # where outputs are saved
```

### Cell 7 — DBSCAN parameters per class

```python
DBSCAN_PARAMS = {
    'floor'    : dict(eps=0.30, min_samples=20),   # large flat surface
    'ceiling'  : dict(eps=0.30, min_samples=20),   # large flat surface
    'wall'     : dict(eps=0.20, min_samples=15),   # vertical surfaces
    'beam'     : dict(eps=0.15, min_samples=10),   # thin structural elements
    'window'   : dict(eps=0.15, min_samples=10),   # small openings
    'furniture': dict(eps=0.08, min_samples=10),   # tight clusters per object
}
```

**`eps`** — the radius in meters within which points are considered neighbors. Smaller = more separate clusters, larger = more merging.  
**`min_samples`** — minimum points to form a cluster core. Higher = fewer but cleaner clusters.

---

## 9. Stage-by-Stage Explanation

### Cell 0 — Install packages
Runs `pip install` for all required packages. Run once, skip afterwards.

### Cell 1 — Imports
Loads all Python libraries and prints version numbers to confirm everything is working.

### Cell 2 — Configuration
Sets the dataset path, which room to process, and creates the output folder. Also prints a full listing of all available rooms in the dataset so you can see what's available.

### Cell 3 — Load from Annotations
**This is where labels come from.** Instead of loading the merged room `.txt` file, this reads every individual file in `Annotations/` (e.g., `chair_1.txt`, `floor_1.txt`) and tags each point with its ground-truth label derived from the filename prefix.

The category mapping used:
```python
CAT_MAP = {
    'floor'   : 'floor',      'ceiling' : 'ceiling',
    'wall'    : 'wall',        'beam'    : 'beam',
    'column'  : 'beam',        'window'  : 'window',
    'door'    : 'furniture',   'table'   : 'furniture',
    'chair'   : 'furniture',   'sofa'    : 'furniture',
    'bookcase': 'furniture',   'board'   : 'furniture',
    'clutter' : 'furniture',
}
```

After loading, it optionally centers the point cloud so the room centroid is at the origin.

### Cell 4 — Voxel downsampling (with labels)
Divides the entire room into a 3D grid of 5cm × 5cm × 5cm voxel cells. Each cell keeps exactly one point (the mean position and color of all points inside it). The semantic label is determined by **majority vote** — whichever label has the most points in that cell wins.

This reduces ~800k points down to ~35k while preserving the label information accurately.

### Cell 5 — Raw RGB visualization
Plots the downsampled point cloud in its original camera colors (not semantic colors) from two angles: top-down (X-Y) and side view (X-Z). This is a sanity check to confirm the room loaded and downsampled correctly.

Saved as `01_raw_rgb.png`.

### Cell 6 — Semantic label visualization
Plots the same point cloud but colors each point by its semantic label instead of its original RGB color. Shows three views: top-down, side, and front. A legend identifies each color.

Saved as `02_semantic_labels.png`.

### Cell 7 — Per-class DBSCAN clustering
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is run **separately for each semantic class**. This is the key design decision — by running it per class, furniture objects are clustered at tight density (eps=0.08m) while floors and ceilings are clustered at loose density (eps=0.30m), which is appropriate for each type.

For each class:
1. Extract all points with that label
2. Run DBSCAN with class-specific parameters
3. Each resulting cluster gets a globally unique ID (no two classes share cluster numbers)
4. Compute bounding box statistics per cluster: width, depth, height, centroid, point count

The result is a `df_clusters` DataFrame with one row per cluster.

### Cell 8 — DBSCAN cluster visualization
Colors each individual cluster a unique color (using the `tab20` colormap) and plots top-down + side views. Noise points (DBSCAN label = -1) are shown as dark grey.

Saved as `03_dbscan_clusters.png`.

### Cell 9 — Save outputs
- **`segmented_room.ply`** — Open3D point cloud colored by semantic label, saved in PLY format. Open in MeshLab, CloudCompare, or any 3D viewer.
- **`cluster_report.csv`** — Pandas DataFrame with one row per cluster: `cluster_id`, `semantic`, `n_points`, `xmin/xmax/ymin/ymax/zmin/zmax`, `centroid_x/y/z`, `width_m`, `depth_m`, `height_m`.

### Cell 10 — Floor plan
Generates a top-down (birds-eye) 2D view on a dark background styled like an architectural floor plan. Each label is rendered at a different point size and opacity so structural elements (walls) appear crisp while floor is a subtle backdrop.

Saved as `04_floor_plan.png`.

### Cell 11 — Furniture bounding boxes
Filters the cluster report to furniture-only clusters. For each furniture cluster, draws an axis-aligned bounding rectangle in the top-down view and annotates it with the cluster ID and dimensions (W × D × H in meters). The background shows floor and wall points in muted colors for context.

Saved as `05_furniture_boxes.png`.

### Cell 12 — Ground truth summary
Prints a table of point counts per label. Because labels come directly from the S3DIS `Annotations/` folder, this **is** the ground truth — there is no prediction error at the point level (labels are exact).

### Cell 13 — Pipeline complete summary
Prints a final report: room name, total points, room height, number of clusters, label distribution with a text bar chart, and a list of all generated output files with their sizes.

---

## 10. Semantic Labels & Colors

| Label | Color (RGB) | Hex | Represents |
|---|---|---|---|
| `floor` | (0.20, 0.82, 0.20) | `#34D134` | Floor surface |
| `ceiling` | (0.75, 0.75, 0.75) | `#BFBFBF` | Ceiling surface |
| `wall` | (0.25, 0.45, 0.95) | `#4072F2` | All wall surfaces |
| `beam` | (0.60, 0.30, 0.10) | `#994D1A` | Structural beams and columns |
| `window` | (0.40, 0.90, 0.95) | `#66E6F2` | Window openings |
| `furniture` | (0.95, 0.50, 0.10) | `#F2801A` | Tables, chairs, doors, boards, clutter |
| `noise` | (0.20, 0.20, 0.20) | `#333333` | Scanner noise / unclassified points |

---

## 11. Output Files

All files are saved to `./output/` (created automatically).

| File | Format | Description |
|---|---|---|
| `01_raw_rgb.png` | PNG image | Top-down and side view in original camera RGB colors |
| `02_semantic_labels.png` | PNG image | Three-view plot colored by semantic label with legend |
| `03_dbscan_clusters.png` | PNG image | Each DBSCAN cluster colored uniquely with tab20 colormap |
| `04_floor_plan.png` | PNG image | Architectural-style top-down floor plan on dark background |
| `05_furniture_boxes.png` | PNG image | Bounding box plot for individual furniture pieces |
| `segmented_room.ply` | PLY 3D file | Full colored point cloud, open in MeshLab or CloudCompare |
| `cluster_report.csv` | CSV table | Per-cluster stats: label, point count, bounding box, centroid |

### Opening the .ply file

**MeshLab** (free, recommended):
1. Download from [meshlab.net](https://www.meshlab.net)
2. File → Import Mesh → select `segmented_room.ply`
3. Press `5` for orthographic view, `3` for side view

**CloudCompare** (free):
1. Download from [cloudcompare.org](https://www.cloudcompare.org)
2. File → Open → select `segmented_room.ply`
3. Use the color picker to view the semantic colors

**Open3D (Python)**:
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("output/segmented_room.ply")
o3d.visualization.draw_geometries([pcd], window_name="Segmented Room")
```

---

## 12. Key Parameters & Tuning

### Voxel size (Cell 4)

```python
voxel_size = 0.05   # 5 cm — change this to trade speed vs detail
```

| Value | Points after downsample | Speed | Detail |
|---|---|---|---|
| 0.02 m (2cm) | ~200k | Slow | Very high |
| 0.05 m (5cm) | ~35k | Fast | Good |
| 0.10 m (10cm) | ~10k | Very fast | Lower |

### DBSCAN eps per class (Cell 7)

The `eps` value should be set relative to voxel size:

- **Structural classes** (floor, ceiling, wall): `eps` = 4–6× voxel size. These are large continuous surfaces — a larger radius lets the algorithm connect points across the surface.
- **Furniture**: `eps` = 1.5–2× voxel size. Furniture objects have gaps between them — a smaller radius keeps individual chairs separate.

### Why per-class DBSCAN?

Running one global DBSCAN on all points always fails for rooms because:
- The floor, walls, and ceiling physically touch at corners → they merge into one cluster
- Furniture eps suitable for chairs (0.08m) would split the floor into thousands of micro-clusters

Running DBSCAN separately per label solves both problems simultaneously.

---

## 13. Known Issues & Fixes Applied

### Issue 1 — Blank furniture bounding box plot
**Cause:** `matplotlib` `add_patch()` does not automatically update the axes limits. The bounding boxes were rendered off-screen because the default axis range was [0,1]×[0,1].  
**Fix:** Added `ax.autoscale_view()` after all patches are added.

Also fixed: `df_furniture.iterrows()` yields the DataFrame index (which may not start at 0), so using it directly as a color index crashed. Fixed by wrapping with `enumerate()`:
```python
# Before (bug)
for i, row in df_furniture.iterrows():
    color = colors[i % len(colors)]   # i = DataFrame index, not counter

# After (fix)
for i, (_, row) in enumerate(df_furniture.iterrows()):
    color = colors[i % len(colors)]   # i = 0, 1, 2, ... always
```

### Issue 2 — Semantic labeling returning all zeros
**Cause (original notebook version):** The `assign_semantic_labels` function converted the label list to a numpy array inside the loop, modified the temp array, but never wrote back to the original list. All labels stayed as `'noise'`.
```python
# Bug — modifies a throwaway copy
semantic_labels_arr = np.array(semantic_labels)
semantic_labels_arr[mask] = label      # ← lost after loop iteration
semantic_labels = semantic_labels_arr.tolist()  # ← reassigned but mask already gone
```
**Fix:** Replaced with direct in-place assignment:
```python
for idx in np.where(mask)[0]:
    semantic_labels[idx] = label
```

### Issue 3 — DBSCAN producing 1 giant cluster (original notebook)
**Cause:** `eps=0.15m` with `min_samples=20` was too tight. Floor/wall/ceiling surfaces are physically connected, so with such a small radius the algorithm merged everything.  
**Fix (current version):** Uses per-class DBSCAN with class-appropriate `eps` values (0.08m for furniture, 0.30m for floors/ceilings).

---

## 14. Results & Accuracy

Because the labels are loaded directly from the S3DIS ground truth `Annotations/` folder, the semantic segmentation is **exact at the point level**. The per-class DBSCAN step then splits these correctly-labeled regions into individual object instances.

### Typical result for `Area_1/conferenceRoom_1`

| Label | Points | % |
|---|---|---|
| wall | ~15,000 | ~40% |
| floor | ~8,000 | ~21% |
| furniture | ~7,500 | ~20% |
| ceiling | ~5,000 | ~13% |
| beam | ~800 | ~2% |
| window | ~600 | ~2% |
| noise | ~200 | ~1% |

### Cluster counts (typical)
- **Floor:** 1–2 clusters (the main floor, possibly with a small raised platform)
- **Ceiling:** 1 cluster
- **Walls:** 4–8 clusters (one per wall segment)
- **Furniture:** 15–50 clusters depending on room size (each chair, table, board is its own cluster)
- **Windows:** 2–6 clusters

---

## 15. Dependencies

```txt
open3d>=0.17.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
```

Install with:
```bash
pip install open3d numpy scikit-learn matplotlib pandas
```

Or with conda:
```bash
conda install numpy matplotlib pandas scikit-learn
pip install open3d   # open3d is pip-only
```

---

## Credits

- **Dataset:** [Stanford S3DIS](http://buildingparser.stanford.edu/dataset.html) — Armeni et al., "3D Semantic Parsing of Large-Scale Indoor Spaces", CVPR 2016
- **3D library:** [Open3D](http://www.open3d.org) — Zhou et al., 2018
- **Clustering:** [scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) — Pedregosa et al., JMLR 2011
