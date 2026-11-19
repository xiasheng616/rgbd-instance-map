from pathlib import Path
import json
from arkit_io import associate_frames,read_intrinsics,read_trajectory
import cv2 as cv
import numpy as np
from geometry import transform_points
import open3d as o3d



mask_dir = Path(r"E:\shixi\rgbd_instance_map\outputs\stage03_bed_masks")
output_dir = Path("outputs/stage03_bed_observations")

with (mask_dir / "results.json").open("r", encoding="utf-8") as f:
    report = json.load(f)

scene_dir = Path(
    r"E:\shixi\rgbd_instance_map\data\arkitscenes\raw\Training\47333462"
)

pose_timestamps, T_wc_all = read_trajectory(
    scene_dir / "lowres_wide.traj"
)
records = associate_frames(scene_dir, pose_timestamps)

records_by_name = {
    record["rgb_path"].name: record for record in records
}

results = []

for image_info in report["images"]:

    image_path = Path(image_info["image"])
    image_name = image_path.name
    record = records_by_name[image_name]

    for object_id,detection in enumerate(image_info["detections"]):

        mask_path = mask_dir / detection["mask_file"]
        mask_gray = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        if mask_gray is None:
            raise OSError(f"无法读取掩码：{mask_path}")

        mask = mask_gray > 0

        K, width, height = read_intrinsics(record["intrinsics_path"])

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

        assert mask.shape == depth.shape == confidence.shape

        v, u = np.indices(depth.shape)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        Z = depth / 1000.0
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        valid = mask & (Z > 0) & (Z < 5.0) & (confidence != 0)

        if not valid.any():

            print(image_name)
            continue


        points_camera = np.stack((X, Y, Z), axis=-1)[valid]
        colors = rgb[valid].astype(np.float64) / 255.0


        T_wc = T_wc_all[record["pose_index"]]
        points_world = transform_points(points_camera, T_wc)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(voxel_size=0.02)

        saved_points = np.asarray(pcd.points)

        centroid = saved_points.mean(axis=0)
        bbox_min = saved_points.min(axis=0)
        bbox_max = saved_points.max(axis=0)

        results.append({
            "frame_id": image_path.stem,
            "local_id": detection["object_id"],
            "label": detection["label"],
            "detection_score": detection["detection_score"],
            "point_count": saved_points.shape[0],
            "centroid": centroid.tolist(),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "cloud_file": f"{image_path.stem}_{object_id}.ply",
        })

        output_dir.mkdir(parents=True, exist_ok=True)

        if not o3d.io.write_point_cloud(str(output_dir / f"{image_path.stem}_{object_id}.ply"), pcd):
            raise OSError("点云保存失败")

        o3d.visualization.draw_geometries([pcd], window_name="Reference")



observations_path = output_dir / "observations.json"
observations_path.parent.mkdir(parents=True, exist_ok=True)
with observations_path.open("w", encoding="utf-8") as f:
    json.dump({
        "scene_id": scene_dir.name,
        "coordinate_frame": "world",
        "observations": results,
    }, f, ensure_ascii=False, indent=2)





