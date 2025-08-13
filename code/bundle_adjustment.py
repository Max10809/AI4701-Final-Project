# bundle_adjustment.py
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def rotate(points: np.ndarray, rot_vecs: np.ndarray) -> np.ndarray:
    """
    Rotate points by given rotation vectors using Rodrigues' formula.
    points:    (N,3)
    rot_vecs:  (N,3)
    returns:   (N,3) rotated points
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    # avoid division by zero
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return cos_t * points + sin_t * np.cross(v, points) + dot * (1 - cos_t) * v

def project(points: np.ndarray,
            camera_params: np.ndarray) -> np.ndarray:
    """
    Project 3D points into 2D by applying rotation and translation,
    then dividing by depth.
    points:          (M,3)
    camera_params:   (M,6) each row [rvec (3), tvec (3)]
    returns:         (M,2) projected points (normalized image coords)
    """
    # apply rotation
    pts_rot = rotate(points, camera_params[:, :3])
    # apply translation
    pts_trans = pts_rot + camera_params[:, 3:6]
    # perspective divide
    return pts_trans[:, :2] / pts_trans[:, 2, np.newaxis]

def fun(params: np.ndarray,
        n_cameras: int,
        n_points: int,
        camera_indices: np.ndarray,
        point_indices: np.ndarray,
        points_2d: np.ndarray,
        K: np.ndarray) -> np.ndarray:
    """
    Compute residuals for bundle adjustment.
    params:           vector of length 6*n_cameras + 3*n_points
    camera_indices:   (m,) which camera observed each point
    point_indices:    (m,) which 3D point is observed
    points_2d:        (m,2) observed pixel coords
    K:                (3,3) intrinsic matrix
    returns:          (2*m,) residual vector
    """
    # unpack camera parameters and 3D points
    cams = params[:n_cameras * 6].reshape((n_cameras, 6))
    pts3d = params[n_cameras * 6:].reshape((n_points, 3))

    # reproject points
    cam_params_obs = cams[camera_indices]
    pts3d_obs = pts3d[point_indices]
    proj = project(pts3d_obs, cam_params_obs)  # (m,2)

    # convert to normalized coords before comparing
    # assume points_2d given in pixels; convert to normalized by inv(K)
    # here points_2d already pre-normalized by caller
    # compute residual and weight by focal
    residual = proj - points_2d  # (m,2)
    # optionally scale by focal lengths:
    residual = (K[0,0] * residual[:,0], K[1,1] * residual[:,1])
    residual = np.stack(residual, axis=1)
    return residual.ravel()

def bundle_adjustment_sparsity(n_cameras: int,
                               n_points: int,
                               camera_indices: np.ndarray,
                               point_indices: np.ndarray) -> lil_matrix:
    """
    Build the sparsity matrix for the Jacobian.
    returns: (2*m, 6*n_cameras + 3*n_points) sparse pattern
    """
    m = camera_indices.size
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((2 * m, n), dtype=int)
    i = np.arange(m)
    # camera params
    for s in range(6):
        A[2*i, camera_indices * 6 + s] = 1
        A[2*i+1, camera_indices * 6 + s] = 1
    # point params
    for s in range(3):
        A[2*i, n_cameras*6 + point_indices * 3 + s] = 1
        A[2*i+1, n_cameras*6 + point_indices * 3 + s] = 1
    return A

def bundle_adjustment(camera_params: np.ndarray,
                      points_2d: np.ndarray,
                      points_3d: np.ndarray,
                      camera_indices: np.ndarray,
                      point_indices: np.ndarray,
                      intrinsic: np.ndarray) -> 'OptimizeResult':
    """
    Perform bundle adjustment to refine camera extrinsics and 3D points.

    camera_params:   (n_cameras,6) initial [rvec, tvec]
    points_2d:       (m,2) observed pixel coords
    points_3d:       (n_points,3) initial 3D locations
    camera_indices:  (m,) camera index for each observation
    point_indices:   (m,) point index for each observation
    intrinsic:       (3,3) camera intrinsic matrix
    """
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    # convert 2D points to normalized camera coordinates
    # append homogeneous coordinate
    if points_2d.shape[1] == 2:
        homo = np.ones((points_2d.shape[0], 1))
        pts_h = np.hstack((points_2d, homo))
    else:
        pts_h = points_2d
    invK = np.linalg.inv(intrinsic)
    pts_norm = (invK @ pts_h.T).T[:, :2]

    # pack initial parameters
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # build sparsity pattern
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    # solve
    res = least_squares(
        fun, x0,
        jac_sparsity=A,
        verbose=2,
        x_scale='jac',
        ftol=1e-3,
        method='trf',
        loss='soft_l1',
        args=(n_cameras, n_points, camera_indices, point_indices, pts_norm, intrinsic)
    )
    return res
