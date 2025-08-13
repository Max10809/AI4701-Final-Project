"""initial_recon.py — 基于对极几何的场景初始化
================================================
读取真实相机内参矩阵 *camera_intrinsic.txt* ，匹配特征→估计本质矩阵
→ 恢复相机相对位姿 → 三角化得到第一批 3‑D 点。

用法
----
$ python initial_recon.py images/0001.jpg images/0002.jpg \
                      --K camera_intrinsic.txt \
                      --ratio 0.75

参数说明
~~~~~~~~
img1, img2 : 两张待初始化的图像路径。
--K        : 3×3 内参文本文件路径，默认为当前目录 camera_intrinsic.txt。
--ratio    : Lowe ratio 阈值，传给 match_features 的 ratio_test，可调。

依赖: opencv‑contrib‑python >= 4.x, numpy
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from feature_matching import match_features

###############################################################################
# 内参读取
###############################################################################

def load_intrinsic(file_path: str | Path) -> np.ndarray:
    """读取 3×3 相机内参矩阵 K"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"[initial_recon] 相机内参文件不存在: {file_path}")

    K = np.loadtxt(file_path, dtype=float)
    if K.shape != (3, 3):
        raise ValueError("camera_intrinsic.txt 格式应为 3×3 浮点数矩阵")
    return K

###############################################################################
# 对极几何核心函数
###############################################################################

def compute_relative_pose(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """估计 Essential 矩阵并恢复相机相对位姿 (R, t)。返回 R, t, inlier_mask"""
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        raise RuntimeError("findEssentialMat 失败，可能匹配点过少或误差过大")

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, mask_pose  # t: (3,1)


def triangulate_points(R: np.ndarray, t: np.ndarray, pts1: np.ndarray,
                       pts2: np.ndarray, K: np.ndarray) -> np.ndarray:
    """三角化 inlier 匹配点，返回 Nx3 实坐标"""
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    pts1_h = pts1.T
    pts2_h = pts2.T
    pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d

###############################################################################
# 主程序
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="对两张图片做初始双目重建")
    parser.add_argument("img1", nargs="?", default="images/0001.jpg")
    parser.add_argument("img2", nargs="?", default="images/0002.jpg")
    parser.add_argument("--K", type=str, default="camera_intrinsic.txt",
                        help="相机内参文件路径 (3×3 文本)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Lowe ratio 阈值 (传给 match_features 的 ratio_test)")
    args = parser.parse_args()

    # 1) 读取内参
    K = load_intrinsic(args.K)
    print("[INFO] 使用的相机内参 K:\n", K)

        # 2) 匹配特征
    good_matches, pts1, pts2 = match_features(args.img1, args.img2)
    print(f"[INFO] good matches: {len(good_matches)}")

    if len(good_matches) < 8:
        raise RuntimeError("匹配点过少 (<8)，无法估计本质矩阵")

    pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
    pts2 = np.ascontiguousarray(pts2, dtype=np.float64)

    # 3) 估计 R, t
    R, t, inlier_mask = compute_relative_pose(pts1, pts2, K)
    print("[INFO] R =\n", R)
    print("[INFO] t =\n", t.ravel())
    print(f"[INFO] pose inliers: {int(inlier_mask.sum())}/{len(good_matches)}")

    # 4) 三角化
    pts3d = triangulate_points(R, t, pts1[inlier_mask.ravel() == 1],
                               pts2[inlier_mask.ravel() == 1], K)
    print(f"[INFO] triangulated {pts3d.shape[0]} 3‑D points")

    # 保存结果
    np.save("init_points.npy", pts3d)
    np.savez("init_pose.npz", R=R, t=t)
    print("[INFO] 3‑D 点与相机姿态已保存 (init_points.npy, init_pose.npz)")


###############################################################################
if __name__ == "__main__":
    main()
