# AI4701-Final-Project

## Introduction
Final project for the AI4701 Computer Vision course at SJTU.
I implement 3D reconstruction of an indoor Triceratops scene from multi-view images using SIFT feature matching and PnP-based pose estimation, followed by triangulation to recover a sparse 3D structure. See report.pdf for detailed methodology and experiments.

## Reproducible outputs
recon.ply — reconstructed sparse 3D point cloud

init_points.npy — 3D points (NumPy)

init_pose.npz — camera poses (NumPy)

intrinsic_matrix.txt — camera intrinsic matrix (K)

## Repository Structure
```
.
├── code/                  # Source code
├── images/                # Input multi-view images of the indoor scene
├── matches/               # (Optional) cached matches/visualizations
├── SIFT/                  # (Optional) cached keypoints/descriptors
├── init_points.npy        # Reconstructed 3D points (NumPy array)
├── init_pose.npz          # Camera poses/extrinsics (NumPy arrays)
├── intrinsic_matrix.txt   # 3×3 camera intrinsic matrix K
├── recon.ply              # Output sparse point cloud
├── report.pdf             # Full report
└── README.md
```
## How to run
First download all the python requirements:
```
- Python 3.8+
- numpy
- opencv-contrib-python
- matplotlib
- scipy
- open3d
- tqdm
pip install numpy opencv-contrib-python matplotlib scipy open3d tqdm
```
Paths below assume you are at the repo root.
### 1) Minimal PnP reconstruction (no BA)
`python code/pnp_recon.py`
### 2)Full reconstruction script
`python code/3d_recon.py`
