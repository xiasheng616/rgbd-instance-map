from scipy.spatial.transform import Rotation
import numpy as np



def pose_to_matrix(pose):
    """输入 (7,) 位姿，返回 (4, 4) 的 T_wc。"""
    T = np.eye(4)
    R = Rotation.from_quat(pose[3:7]).as_matrix()
    T[:3,:3]=R
    T[:3, 3] = pose[:3]

    return T



def transform_points(points, T):
    """输入 (N, 3) 点坐标和 (4, 4) 变换，返回变换后的 (N, 3) 点坐标。"""
    R = T[:3, :3]
    t = T[:3, 3]

    points_world = points @ R.T + t

    return points_world