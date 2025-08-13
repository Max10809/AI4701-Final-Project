#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pnp_recon.py  ——  增量式 PnP-SfM（无 Bundle Adjustment）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
流程概要
---------
1. 读取图像 → 通过 feature_matching._get_features **一次性** 缓存 SIFT 特征。
2. 以 0001–0002 为初始对：
      • 估计 Essential → recoverPose → Triangulation
3. 第 3 张开始：
      • 从已有 3-D 取 2D-3D 对应 → PnP-RANSAC 定位
      • 与前 ≤4 帧且视差角≥2° 的帧做 **新特征** 三角化
      • 三角化后立刻按重投影误差 (<3 px) 过滤
4. 全部结束后，用 Open3D **统计离群滤波** 清理噪点
5. 保存 model_pnp.ply

依赖
-----
pip install numpy opencv-python open3d tqdm
（feature_extraction/feature_matching 已由你提供）

*完全不调用 Bundle Adjustment，整套 62 张图 < 3 GB 内存即可完成。*
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from feature_matching import match_features, _get_features
from initial_recon   import triangulate_points

from bundle_adjustment import bundle_adjustment

# ------------------- 全局超参数 -------------------
PAIR_WIN       = 6          # 每帧向前可用于三角化的参考帧数
PARALLAX_DEG   = 1.0        # 最小视差角 (deg) — 小于该值不三角化
REPROJ_ERR_TH  = 1.0        # 三角化后重投影阈值 (px)
MIN_PNP_PTS    = 20         # 运行 PnP 的最小 2D-3D 对
# --------------------------------------------------

VOXEL_SIZE    = 0.01  # 体素下采样大小 (世界单位)，可调，越大点越少
STAT_NB       = 30    # 统计滤波：每点邻居数
STAT_RATIO    = 1.5   # 统计滤波：标准差倍数
RADIUS_NB     = 10    # 半径滤波：最少邻居数
RADIUS_SCALE  = 0.1  # 半径滤波：基于点云最大边长的比例
CLUSTER_EPS   = 0.2  # DBSCAN 聚类参数：邻域半径
CLUSTER_MIN   = 3    # DBSCAN 聚类参数：最小点数
SPHERE_SCALE  = 0.1   # 基于最大簇半径裁剪比率
# ---------- 工具函数 ----------
def pose_to_proj(K, R, t):
    return K @ np.hstack([R, t])


def world_from_cam(X_cam, R, t):
    return (R.T @ (X_cam.T - t)).T


def calc_parallax(R1, R2):
    """相机光轴夹角（deg，用于判断基线长度）"""
    cosang = np.clip(np.dot(R1[2], R2[2])
                     / (np.linalg.norm(R1[2]) * np.linalg.norm(R2[2])), -1, 1)
    return math.degrees(math.acos(cosang))


def filter_triangulation(R1, t1, R2, t2, X, uv1, uv2, K,
                         th=REPROJ_ERR_TH):
    """根据双向重投影误差滤除离群 3-D 点"""
    P1, P2 = pose_to_proj(K, R1, t1), pose_to_proj(K, R2, t2)
    X_h    = np.hstack([X, np.ones((len(X), 1))]).T
    uv1p   = (P1 @ X_h).T
    uv1p   = uv1p[:, :2] / uv1p[:, 2:3]
    uv2p   = (P2 @ X_h).T
    uv2p   = uv2p[:, :2] / uv2p[:, 2:3]
    err    = np.linalg.norm(uv1p - uv1, axis=1) + np.linalg.norm(uv2p - uv2, axis=1)
    keep   = err < th
    return X[keep], keep


# ---------- 主流程 ----------
def run_pnp_sfm(img_paths: list[str], K: np.ndarray):
    n_img = len(img_paths)
    print(f"[INFO] 共 {n_img} 张图，开始 PnP-SfM（无 BA）")

    # 1. 预加载 / 缓存所有 SIFT 特征（使用 feature_matching 内部缓存）
    kps_all = []
    for p in tqdm(img_paths, desc="加载 SIFT 特征"):
        kps, _ = _get_features(p)      # 与 match_features 保持完全一致顺序
        kps_all.append(kps)
    kps_xy = [np.asarray([kp.pt for kp in kps], dtype=np.float64)
              for kps in kps_all]

    # 结构化存储
    poses: list[tuple[np.ndarray, np.ndarray]] = []  # 每张相机的 (R, t)
    kp_map = [dict() for _ in range(n_img)]          # 每图: kp_idx -> global pt id
    pts3d, colors = [], []                           # 全局 3-D
    matches_dict = [defaultdict(list) for _ in range(n_img)]

    # ------------------------------------------------------------------
    # 2. 用首对（0001-0002）初始化世界坐标
    print("[INIT] 0001 ↔ 0002 ...")
    good, pts1, pts2 = match_features(img_paths[0], img_paths[1])
    if len(good) < 8:
        raise RuntimeError("首对匹配不足，无法初始化")

    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    _, R2, t2, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inl = mask_pose.ravel().astype(bool)
    good_inl = [g for g, keep in zip(good, inl) if keep]
    pts1_inl, pts2_inl = pts1[inl], pts2[inl]

    X_cam0 = triangulate_points(R2, t2, pts1_inl, pts2_inl, K)
    X_w    = world_from_cam(X_cam0, np.eye(3), np.zeros((3, 1)))
    col0   = cv2.imread(img_paths[0])
    cols   = col0[pts1_inl[:, 1].astype(int), pts1_inl[:, 0].astype(int), ::-1]

    base = 0
    for idx, (p3d, m) in enumerate(zip(X_w, good_inl)):
        pts3d.append(p3d)
        colors.append(cols[idx])
        kp_map[0][m.queryIdx] = base + idx
        kp_map[1][m.trainIdx] = base + idx
    matches_dict[1][0] = good_inl
    matches_dict[0][1] = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance)
                          for m in good_inl]

    poses.extend([(np.eye(3), np.zeros((3, 1))), (R2, t2)])

    # ------------------------------------------------------------------
    # 3. 增量处理余下帧
    for i in range(2, n_img):
        print(f"\n[FRAME {i+1:04d}] --------------------------")

        # 3-A 与历史帧做匹配（只保留 ±PAIR_WIN 内的，以后要三角化）
        for ref in range(max(0, i - PAIR_WIN), i):
            good, _, _ = match_features(img_paths[ref], img_paths[i])
            matches_dict[i][ref] = good
            matches_dict[ref][i] = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance)
                                     for m in good]

        # 3-B 组 2D-3D 对，PnP-RANSAC 求位姿
        uv, Pw = [], []
        for ref, matches in matches_dict[i].items():
            for m in matches:          # m.queryIdx 在 ref，m.trainIdx 在 i
                pid = kp_map[ref].get(m.queryIdx)
                if pid is not None:    # 已有 3-D
                    uv.append(kps_xy[i][m.trainIdx])
                    Pw.append(pts3d[pid])
        if len(uv) < MIN_PNP_PTS:
            print("  * 有效 2D-3D 对少于阈值，跳过")
            poses.append(poses[-1])
            continue

        ok, rvec, tvec, inl_pnp = cv2.solvePnPRansac(
            np.asarray(Pw, np.float32), np.asarray(uv, np.float32),
            K, None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=3.0, confidence=0.999, iterationsCount=150
        )
        if not ok:
            print("  * PnP 失败")
            poses.append(poses[-1])
            continue
        R_i, _ = cv2.Rodrigues(rvec)
        t_i    = tvec
        poses.append((R_i, t_i))
        print(f"  · PnP 成功 (inliers {len(inl_pnp)}/{len(uv)})")


        # 3-C 与前 ≤PAIR_WIN 帧做三角化（只处理“新特征”）
        new_pts = 0
        for ref in range(max(0, i - PAIR_WIN), i):
            R_ref, t_ref = poses[ref]
            if calc_parallax(R_ref, R_i) < PARALLAX_DEG:
                continue
            matches = matches_dict[i][ref]
            if len(matches) < 8:
                continue

            uv_ref, uv_cur, ref_idx, cur_idx = [], [], [], []
            for m in matches:
                if (kp_map[ref].get(m.queryIdx) is None and
                        kp_map[i].get(m.trainIdx) is None):
                    uv_ref.append(kps_xy[ref][m.queryIdx])
                    uv_cur.append(kps_xy[i][m.trainIdx])
                    ref_idx.append(m.queryIdx)
                    cur_idx.append(m.trainIdx)

            if len(uv_ref) < 8:
                continue

            R_rel = R_i @ R_ref.T
            t_rel = t_i - R_rel @ t_ref
            X_cam = triangulate_points(R_rel, t_rel,
                                       np.asarray(uv_ref), np.asarray(uv_cur), K)
            X_w   = world_from_cam(X_cam, R_ref, t_ref)
            X_w, keep = filter_triangulation(
                R_ref, t_ref, R_i, t_i, X_w,
                np.asarray(uv_ref), np.asarray(uv_cur), K
            )
            if len(X_w) == 0:
                continue

            dists = np.linalg.norm(X_w, axis=1)

            base_id = len(pts3d)
            col_ref = cv2.imread(img_paths[ref])
            cols = col_ref[np.asarray(uv_ref)[keep][:, 1].astype(int),
                           np.asarray(uv_ref)[keep][:, 0].astype(int), ::-1]
            kk = keep.nonzero()[0]
            for k, p3d in enumerate(X_w):
                pts3d.append(p3d)
                colors.append(cols[k])
                kp_map[ref][ref_idx[kk[k]]] = base_id + k
                kp_map[i][cur_idx[kk[k]]]   = base_id + k
            new_pts += len(X_w)
        print(f"  · 新增点数 {new_pts}")

    print("\n[BA] Running bundle adjustment ...")
    n_cameras = len(poses);
    n_points = len(pts3d)
    camera_params = np.zeros((n_cameras, 6))
    for i, (R, t) in enumerate(poses):
        rvec, _ = cv2.Rodrigues(R)
        camera_params[i, :3] = rvec.flatten()
        camera_params[i, 3:] = t.flatten()
    cam_idxs, pt_idxs, pts2d = [], [], []
    for cam_idx in range(n_cameras):
        for kp_idx, pid in kp_map[cam_idx].items():
            cam_idxs.append(cam_idx);
            pt_idxs.append(pid)
            pts2d.append(kps_xy[cam_idx][kp_idx])
    cam_idxs = np.asarray(cam_idxs);
    pt_idxs = np.asarray(pt_idxs);
    pts2d = np.asarray(pts2d)
    ba_res = bundle_adjustment(camera_params, pts2d, np.asarray(pts3d),
                               cam_idxs, pt_idxs, K)
    opt = ba_res.x
    cam_opt = opt[:n_cameras * 6].reshape(n_cameras, 6)
    pts3d_opt = opt[n_cameras * 6:].reshape(n_points, 3)
    poses = []
    for i in range(n_cameras):
        rvec = cam_opt[i, :3];
        R_opt, _ = cv2.Rodrigues(rvec)
        t_opt = cam_opt[i, 3:].reshape(3, 1)
        poses.append((R_opt, t_opt))
    pts3d = [pts3d_opt[i] for i in range(n_points)]
    print("[BA] Bundle adjustment completed")

    print("\n[POST] 点云滤波清理 ...")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(pts3d))
    pc.colors = o3d.utility.Vector3dVector(np.array(colors, float) / 255.0)
    print(f"  原始点数: {len(pc.points)}")
    # 4.1 体素下采样
    pc = pc.voxel_down_sample(VOXEL_SIZE)
    print(f"  下采样后点数: {len(pc.points)}")
    # 4.2 统计离群滤波
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=STAT_NB, std_ratio=STAT_RATIO)
    print(f"  统计滤波后点数: {len(pc.points)}")

    center = pc.get_center();
    pts = np.asarray(pc.points)
    dists = np.linalg.norm(pts - center, axis=1);
    R_max = dists.max() * SPHERE_SCALE
    idxs = np.where(dists <= R_max)[0];
    pc = pc.select_by_index(idxs)
    print(f"  球体裁剪后: {len(idxs)}")

    extrinsics_path = Path("camera_extrinsics.txt")
    R_base, t_base = poses[-1]
    with open(extrinsics_path, 'w') as f:
        # 先写 0062 的外参 (单位阵)
        ext = np.eye(4)
        f.write(' '.join(f"{v:.6f}" for v in ext.flatten()) + "\n")
        # 再写 0001~0061 的外参
        for idx in range(0, n_img - 1):
            R_i, t_i = poses[idx]
            R_rel = R_i @ R_base.T
            t_rel = t_i - R_rel @ t_base
            ext = np.eye(4)
            ext[:3, :3] = R_rel
            ext[:3, 3] = t_rel.flatten()
            f.write(' '.join(f"{v:.6f}" for v in ext.flatten()) + "\n")
    print(f"[SAVE] 相机外参已写入 {extrinsics_path}")

    return pc


def save_ply(path: str | Path, pc: o3d.geometry.PointCloud):
    o3d.io.write_point_cloud(str(path), pc, write_ascii=True)
    print(f"[SAVE] 点云写入 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", type=str, default="images",
                        help="包含 0001.jpg ~ n.jpg 的文件夹")
    parser.add_argument("--num_imgs", type=int, default=62,
                        help="图像张数 (默认 62)")
    parser.add_argument("--K", type=str, default="camera_intrinsic.txt",
                        help="3×3 相机内参")
    parser.add_argument("--out", type=str, default="model.ply",
                        help="输出点云文件名")
    args = parser.parse_args()

    img_paths = [str(Path(args.imgs_dir) / f"{i:04d}.jpg")
                 for i in range(1, args.num_imgs + 1)]
    K = np.loadtxt(args.K, dtype=float)
    if K.shape != (3, 3):
        raise ValueError("camera_intrinsic.txt 应为 3×3 浮点矩阵")

    # ---------- 在 main() 里，拿到 pc 之后 ----------
    pc = run_pnp_sfm(img_paths, K)  # 稀疏点云（含少量远端离群）

    '''
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.array(pc.cluster_dbscan(eps=0.2, min_points=30, print_progress=False))
    if labels.max() >= 0:
        biggest = np.bincount(labels[labels >= 0]).argmax()
        pc = pc.select_by_index(np.where(labels == biggest)[0])
        print(f"[CLEAN] 保留最大簇 {len(pc.points)} 点")
    else:
        print("[CLEAN] DBSCAN 未找到簇，跳过")
    '''

    # 3) 平移到原点
    center = pc.get_center()
    pc.translate(-center)
    print("[CLEAN] 平移质心到原点", center)

    # 4) 归一化最长边到 1，再按需放大 (这里放大 5×)
    bbox = pc.get_axis_aligned_bounding_box()
    scale1 = 1.0 / max(bbox.get_extent())  # 长边→1
    pc.scale(scale1, center=(0, 0, 0))
    EXTRA = 1.0  # 想再放大倍数
    pc.scale(EXTRA, center=(0, 0, 0))
    print(f"[CLEAN] 归一化并额外放大 {EXTRA}×")

    # 5) 保存
    save_ply(args.out, pc)

    # 6) 弹窗查看
    o3d.visualization.draw_geometries([pc],
                                      window_name="PnP-SfM Cleaned", width=1280, height=720)


if __name__ == "__main__":
    main()