from rgbd_map.geometry import pose_to_matrix,transform_points
from rgbd_map.tum_io import read_image_index,read_pose_index,associate_records
import numpy as np
from pathlib import Path
import cv2 as cv
import open3d as o3d
import argparse




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--voxel-size",type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        parser.error("--stride 必须大于 0")

    if args.max_frames <= 0:
        parser.error("--max-frames 必须大于 0")

    if args.voxel_size <= 0:
        parser.error("--voxel-size 必须大于 0")

    return args




if __name__ == "__main__":

    args = parse_args()

    sequence_dir = args.sequence

    rgb_records = read_image_index(sequence_dir / "rgb.txt")
    depth_records = read_image_index(sequence_dir / "depth.txt")
    pose_timestamps,poses = read_pose_index(sequence_dir / "groundtruth.txt")

    rgb_path = sequence_dir / rgb_records[0][1]
    depth_path = sequence_dir / depth_records[0][1]

    matches = associate_records(
            rgb_records,
            depth_records,
            pose_timestamps,
        )
    selected_matches = matches[::args.stride][:args.max_frames]

    merged_pcd=o3d.geometry.PointCloud()

    camera_centers = []

    for index, record in enumerate(selected_matches):

        rgb_path = sequence_dir / rgb_records[record["rgb_index"]][1]
        depth_path = sequence_dir / depth_records[record["depth_index"]][1]
        pose = poses[record["pose_index"]]

        rgb = cv.imread(str(rgb_path))
        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        depth = cv.imread(str(depth_path), cv.IMREAD_UNCHANGED)

        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]

        v, u = np.indices(depth.shape)
        fx = fy = 525.0
        cx = 319.5
        cy = 239.5

        Z = depth / 5000.0
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        valid = (Z > 0) & (Z < 4.0)
        points = np.stack((X, Y, Z), axis=-1)[valid]
        colors = rgb[valid].astype(np.float64) / 255.0

        T_wc = pose_to_matrix(pose)

        points_world = transform_points(points, T_wc)

        camera_centers.append(T_wc[:3, 3].copy())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

        merged_pcd = merged_pcd + pcd

    print("融合下采样前点数：", len(merged_pcd.points))

    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=args.voxel_size)

    print("融合下采样后点数：", len(merged_pcd.points))

    args.output.mkdir(parents=True, exist_ok=True)

    output_path = args.output / "fused_map.ply"

    success = o3d.io.write_point_cloud(str(output_path), merged_pcd)

    if not success:
        raise IOError(f"点云保存失败：{output_path}")

    print("已保存：", output_path)

    lines=[[i,i+1] for i in range(len(camera_centers)-1)]

    trajectory = o3d.geometry.LineSet()

    trajectory.points = o3d.utility.Vector3dVector(
        np.asarray(camera_centers)
    )
    trajectory.lines = o3d.utility.Vector2iVector(lines)

    trajectory.paint_uniform_color([1, 0, 0])

    if args.show:
        o3d.visualization.draw_geometries([merged_pcd, trajectory])






