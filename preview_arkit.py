from arkit_io import associate_frames,read_intrinsics,read_trajectory
from pathlib import Path
import cv2 as cv
import numpy as np
import open3d as o3d



def create_pcd(record,K,confidence) :

    rgb = cv.imread(str(record["rgb_path"]))
    rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)

    depth = cv.imread(
        str(record["depth_path"]), cv.IMREAD_UNCHANGED
    )

    confidence = cv.imread(
        str(record["confidence_path"]), cv.IMREAD_UNCHANGED
    )

    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]

    v, u = np.indices(depth.shape)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    Z = depth / 1000.0
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    valid = (Z > 0) & (Z < 5.0) & (confidence!=0)
    points = np.stack((X, Y, Z), axis=-1)[valid]
    colors = rgb[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd



scene_dir = Path(
    r"E:\shixi\rgbd_instance_map\data\arkitscenes\raw\Training\47333462"
)

pose_timestamps, T_wc_all = read_trajectory(scene_dir / "lowres_wide.traj")

records = associate_frames(scene_dir,pose_timestamps)

selected_records = records[200:300:10]

reference_map = o3d.geometry.PointCloud()

for index, record in enumerate(selected_records) :

    K, width, height = read_intrinsics(record["intrinsics_path"])

    confidence = cv.imread(
        str(record["confidence_path"]), cv.IMREAD_UNCHANGED
    )

    pcd_c = create_pcd(record,K,confidence)

    pcd_w = pcd_c.transform(T_wc_all[record["pose_index"]])

    reference_map = reference_map + pcd_w




reference_map = reference_map.voxel_down_sample(voxel_size=0.02)

output_dir = Path("outputs/stage03_arkit_preview_other10")
output_dir.mkdir(parents=True, exist_ok=True)

if not o3d.io.write_point_cloud(str(output_dir / "reference_map.ply"), reference_map):

    raise OSError("reference_map 保存失败")

o3d.visualization.draw_geometries([reference_map], window_name="reference")