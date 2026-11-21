import numpy as np
from scipy.spatial.transform import Rotation
from rgbd_map.geometry import transform_points
from rgbd_map.registration import estimate_rigid_transform



if __name__ == "__main__":
    rng = np.random.default_rng(42)
    source = rng.uniform(-1.0, 1.0, size=(100, 3))

    R_true = Rotation.from_euler("xy", [45, 30], degrees=True).as_matrix()
    t_true = np.array([0.1, 0.3, -0.4])

    target = source @ R_true.T + t_true

    noise = rng.normal(0.0, 0.002, size=target.shape)
    target_noisy = target + noise

    T_est = estimate_rigid_transform(source, target_noisy)
    aligned = transform_points(source, T_est)

    print("真实的R：",R_true)
    print("真实的t：",t_true)

    print("估计的R：",T_est[:3, :3])
    print("估计的t：",T_est[:3, 3])

    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]

    rotation_error = np.max(np.abs(R_est - R_true))
    translation_error = np.linalg.norm(t_est - t_true)

    print("旋转矩阵最大元素误差：", rotation_error)
    print("平移向量误差的范数：", translation_error, "米")

    rmse_noisy = np.sqrt(np.mean(np.sum((aligned - target_noisy) ** 2, axis=1)))
    print("与 target_noisy 的对应点 RMSE：", rmse_noisy)
    rmse_clean = np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1)))
    print("与 target_clean 的对应点 RMSE：",rmse_clean)
