"""
依赖: OpenCV-contrib-python ≥ 4.x, NumPy, matplotlib
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from feature_extraction import extract_features

IMG_DIR   = Path("images")
IMG_EXT   = ".jpg"
IDX_RANGE = range(1, 63)             
MATCH_DIR = Path("matches")
MATCH_DIR.mkdir(parents=True, exist_ok=True)

_FLANN_INDEX_KDTREE = 1
_FLANN_PARAMS  = dict(algorithm=_FLANN_INDEX_KDTREE, trees=5)
_FLANN_SEARCH  = dict(checks=50)
_RATIO_THRESH  = 0.75                    # Lowe ratio
_RANSAC_THR_PX = 1.5                     # RANSAC 内点阈值 (像素)

_cache: Dict[str, Tuple[List[cv2.KeyPoint], np.ndarray]] = {}

def _get_features(img_path: str | Path):
    """特征提取带缓存，避免重复计算。"""
    img_path = str(img_path)
    if img_path not in _cache:
        _cache[img_path] = extract_features(img_path)
    return _cache[img_path]

def match_features(img1: str | Path,
                   img2: str | Path,
                   ratio: float = _RATIO_THRESH,
                   ransac_thresh: float = _RANSAC_THR_PX):
    """
    Returns
    -------
    good_matches : list[cv2.DMatch]
    pts1, pts2   : ndarray, (N,2) 像素坐标
    """
    key1, des1 = _get_features(img1)
    key2, des2 = _get_features(img2)
    if des1 is None or des2 is None:
        return [], np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    matcher = cv2.FlannBasedMatcher(_FLANN_PARAMS, _FLANN_SEARCH)
    knn = matcher.knnMatch(des1, des2, 2)
                       
    good, pts1, pts2 = [], [], []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
            pts1.append(key1[m.queryIdx].pt)
            pts2.append(key2[m.trainIdx].pt)

    if len(good) < 8:                           
        return good, np.float32(pts1), np.float32(pts2)

    pts1_np = np.float32(pts1)
    pts2_np = np.float32(pts2)
    F, mask = cv2.findFundamentalMat(
        pts1_np, pts2_np, cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=0.999
    )
    if F is None:                               
        return good, pts1_np, pts2_np

    inl = mask.ravel().astype(bool)
    good_inl = [g for g, keep in zip(good, inl) if keep]
    return good_inl, pts1_np[inl], pts2_np[inl]


def visualize_matches(img1: str | Path, img2: str | Path, save: bool = True):
    good, _, _ = match_features(img1, img2)
    if not good:
        print(f"[WARN] 无匹配: {img1} ↔ {img2}")
        return

    im1 = cv2.imread(str(img1))
    im2 = cv2.imread(str(img2))
    key1, _ = _get_features(img1)
    key2, _ = _get_features(img2)

    vis = cv2.drawMatches(im1, key1, im2, key2, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.imshow(vis)
    plt.axis("off")
    plt.tight_layout()

    if save:
        name1 = Path(img1).stem
        name2 = Path(img2).stem
        out = MATCH_DIR / f"{name1}_{name2}.jpg"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"[SAVE] {out}  |  几何内点: {len(good):4d}")
    else:
        plt.show()

if __name__ == "__main__":
    images = [IMG_DIR / f"{i:04d}{IMG_EXT}" for i in IDX_RANGE]

    # 此处仅演示 **邻近图对**，若要所有组合请改双重循环
    for a, b in zip(images[:-1], images[1:]):
        visualize_matches(a, b)
