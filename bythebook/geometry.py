"""Camera geometry in the table frame: rays, plane intersection, triangulation."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rotvec_T(rotvec, t) -> np.ndarray:
    return make_T(Rotation.from_rotvec(rotvec).as_matrix(), t)


def apply_T(T, P) -> np.ndarray:
    P = np.asarray(P, float)
    return P @ T[:3, :3].T + T[:3, 3]


def pixel_ray(cam, uv):
    """Viewing ray of pixel(s) `uv` in the table frame -> (origin (3,), unit directions (N,3))."""
    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(uv, cam.K, cam.dist).reshape(-1, 2)
    d_cam = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    d = d_cam @ cam.T_table_cam[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return cam.T_table_cam[:3, 3].copy(), d


def ray_plane_z(cam, uv, z: float) -> np.ndarray:
    """Intersect the viewing ray(s) of `uv` with the horizontal plane at height z (table frame)."""
    o, d = pixel_ray(cam, uv)
    s = (z - o[2]) / d[:, 2]
    if np.any(s <= 0):
        raise ValueError("ray does not hit the plane in front of the camera")
    return o + s[:, None] * d


def project(cam, P) -> np.ndarray:
    """Table-frame point(s) -> pixel(s) (N,2), with lens distortion."""
    P = np.asarray(P, float).reshape(-1, 3)
    T_cam_table = inv_T(cam.T_table_cam)
    rvec = cv2.Rodrigues(T_cam_table[:3, :3])[0]
    uv, _ = cv2.projectPoints(P, rvec, T_cam_table[:3, 3], cam.K, cam.dist)
    return uv.reshape(-1, 2)


def triangulate(cams, uvs):
    """Least-squares intersection of one pixel per camera.

    Returns (point (3,), reprojection error per camera in px). Two or more cameras.
    """
    if len(cams) < 2:
        raise ValueError("triangulation needs at least two cameras")
    A, b = np.zeros((3, 3)), np.zeros(3)
    for cam, uv in zip(cams, uvs):
        o, d = pixel_ray(cam, uv)
        M = np.eye(3) - np.outer(d[0], d[0])      # projector onto the plane normal to the ray
        A += M
        b += M @ o
    P = np.linalg.solve(A, b)
    err = np.array([np.linalg.norm(project(c, P)[0] - np.asarray(uv, float)) for c, uv in zip(cams, uvs)])
    return P, err


def rigid_fit(A, B) -> np.ndarray:
    """Kabsch: 4x4 T minimising |T A - B| for point sets (N,3)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(R, cb - R @ ca)
