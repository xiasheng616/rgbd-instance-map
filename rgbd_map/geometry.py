from scipy.spatial.transform import Rotation, Slerp
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


def interpolate_pose(timestamp, pose_timestamps, poses):
    """返回查询时刻的 4×4 相机到世界变换 T_wc。"""

    start_time = pose_timestamps[0]
    end_time = pose_timestamps[-1]

    if timestamp < start_time or timestamp > end_time:
        raise ValueError("查询时间超出了参考位姿的时间范围")


    x = np.interp(timestamp, pose_timestamps, poses[:, 0])
    y = np.interp(timestamp, pose_timestamps, poses[:, 1])
    z = np.interp(timestamp, pose_timestamps, poses[:, 2])

    rotations = Rotation.from_quat(poses[:, 3:7])
    slerp = Slerp(pose_timestamps, rotations)
    R = slerp([timestamp]).as_matrix()[0]

    T_wc = np.eye(4)
    T_wc[:3,:3]=R
    T_wc[:3, 3] = [x,y,z]

    return T_wc