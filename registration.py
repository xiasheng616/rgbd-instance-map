import numpy as np
from scipy.spatial import cKDTree
from geometry import transform_points
from scipy.spatial.transform import Rotation



def estimate_rigid_transform(source, target):
    """输入两个 (N, 3) 数组，每行一对对应点。
    返回 (4, 4) 的 T，将 source 变换到 target。
    """
    c_source=np.mean(source,axis=0)
    c_target=np.mean(target,axis=0)

    P=source-c_source
    Q=target-c_target

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    V = Vt.T
    R = V @ U.T

    if np.linalg.det(R) < 0 :
        Vt[-1,:]=-Vt[-1,:]
        V = Vt.T
        R = V @ U.T

    t = c_target - R @ c_source

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T



def icp_point_to_point(
    source,
    target,
    max_iterations=50,
    max_distance=0.2,
    tolerance=1e-6,
):
    """返回最终 source→target 变换 T，以及每轮 RMSE 列表 history。"""

    history=[]
    epochs = 0
    diff_rmse=float('inf')
    last_rmse = float("inf")

    T = np.eye(4)

    tree = cKDTree(target)

    while epochs < max_iterations and diff_rmse > tolerance :

        current_source = transform_points(source, T)

        distances, indices = tree.query(current_source, k=1)

        valid = distances <= max_distance
        num_matches = np.count_nonzero(valid)

        if num_matches < 3 :

            raise RuntimeError(
                f"ICP 配准失败：有效对应点只有 {num_matches} 对，至少需要 3 对。"
            )


        delta_T = estimate_rigid_transform(
            current_source[valid],
            target[indices[valid]],
        )

        T = delta_T @ T

        rmse = np.sqrt(np.mean(distances[valid] ** 2))

        diff_rmse=abs(last_rmse-rmse)

        last_rmse=rmse

        history.append(rmse)

        # print("epoch :",epochs,"更新前 RMSE :",rmse)

        epochs += 1

    return  T,history


def evaluate_alignment(
    source_points,
    target_points,
    T,
    max_distance=0.10,
):
    """返回有效对应数、fitness 和 inlier_rmse。"""
    source_points=transform_points(source_points,T)

    tree = cKDTree(target_points)

    distances, indices = tree.query(source_points, k=1)

    valid = distances <= max_distance

    num_matches = np.count_nonzero(valid)

    if num_matches == 0:
        raise RuntimeError("配准评价失败：距离阈值内没有有效对应点")

    fitness = num_matches / len(source_points)
    inlier_rmse = np.sqrt(np.mean(distances[valid] ** 2))

    return num_matches,fitness,inlier_rmse


def evaluate_pose(T_est, T_ref):
    """返回旋转角误差（度）、平移误差（米）。"""

    R_delta = T_est[:3, :3] @ T_ref[:3, :3].T

    angle_rad = Rotation.from_matrix(R_delta).magnitude()
    rotation_error_deg = np.degrees(angle_rad)

    translation_error_m = np.linalg.norm(
        T_est[:3, 3] - T_ref[:3, 3]
    )

    return rotation_error_deg,translation_error_m