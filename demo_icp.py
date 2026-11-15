import numpy as np
from scipy.spatial.transform import Rotation
from registration import icp_point_to_point
from geometry import transform_points


if __name__ == "__main__":

    rng = np.random.default_rng(42)
    source = rng.uniform(-1.0, 1.0, size=(1000, 3))

    R_true = Rotation.from_euler(
        "xyz", [3, -2, 4], degrees=True
    ).as_matrix()
    t_true = np.array([0.03, -0.02, 0.01])

    target_clean = source @ R_true.T + t_true
    target = target_clean[rng.permutation(len(target_clean))]

    T_est, history = icp_point_to_point(source, target)

    aligned = transform_points(source, T_est)

    rmse_true = np.sqrt(
        np.mean(np.sum((aligned - target_clean) ** 2, axis=1))
    )

    print("迭代次数：", len(history))
    print("估计变换：\n", T_est)
    print("真实对应点 RMSE：", rmse_true)

