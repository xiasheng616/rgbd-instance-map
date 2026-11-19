import numpy as np
from scipy.spatial.transform import Rotation


def read_intrinsics(path):
    """返回 K: (3,3)，以及整数 width、height。即 K, width, height。"""

    K = np.eye(3)

    with open(path, "r", encoding="utf-8") as f:
        for line in f :
            line=line.strip()
            if not line or line.startswith("#"):
                continue

            width,height,fx,fy,cx,cy = line.split()

            K[0,0],K[1,1],K[0,2],K[1,2]= fx,fy,cx,cy

            width, height = int(width), int(height)

    return K, width, height






def read_trajectory(path):
    """返回 timestamps: (N,)，T_wc_all: (N,4,4)，按时间排序。"""

    timestamps = []
    rows = []
    T_wc_all=[]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            T_cw = np.eye(4)

            timestamp, rx, ry, rz, tx, ty, tz= line.split()

            timestamps.append(float(timestamp))

            R_cw = Rotation.from_rotvec(
                [float(rx), float(ry), float(rz)]
            ).as_matrix()

            T_cw[:3, :3] = R_cw

            T_cw[:3, 3] = [tx, ty, tz]

            T_wc = np.linalg.inv(T_cw)

            T_wc_all.append(T_wc)

    timestamps = np.asarray(timestamps, dtype=np.float64)

    T_wc_all = np.asarray(T_wc_all, dtype=np.float64)

    order = np.argsort(timestamps)

    return timestamps[order], T_wc_all[order]





def associate_frames(scene_dir, pose_timestamps, max_diff=0.005) :

    records = []

    depth_paths = sorted((scene_dir / "lowres_depth").glob("*.png"))

    for depth_path in depth_paths:

        rgb_path = scene_dir / "lowres_wide" / depth_path.name

        confidence_path = scene_dir / "confidence" / depth_path.name

        intrinsics_path= scene_dir / "lowres_wide_intrinsics" / (depth_path.stem + ".pincam")

        if not (
                rgb_path.is_file()
                and confidence_path.is_file()
                and intrinsics_path.is_file()
        ):
            print("缺少配套文件：", depth_path.name)
            continue

        timestamp = float(depth_path.stem.split("_")[1])

        differences = np.abs(pose_timestamps - timestamp)

        pose_index = np.argmin(differences)

        pose_dt=differences[pose_index]

        if pose_dt < max_diff:

            records.append({
                "timestamp": timestamp,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "intrinsics_path": intrinsics_path,
                "confidence_path": confidence_path,
                "pose_index" : pose_index,
                "pose_dt" : pose_dt
            })

    return records