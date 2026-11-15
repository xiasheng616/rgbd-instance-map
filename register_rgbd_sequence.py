import copy
from registration import icp_point_to_point,evaluate_pose
from geometry import interpolate_pose,transform_points
from tum_io import read_image_index,read_pose_index,associate_records
import cv2 as cv
import numpy as np
from pathlib import Path
import open3d as o3d
import argparse
import csv



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--voxel-size",type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--max-distance", type=float, default=0.10)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    if args.stride <= 0:
        parser.error("--stride 必须大于 0")

    if args.max_frames <= 0:
        parser.error("--max-frames 必须大于 0")

    if args.voxel_size <= 0:
        parser.error("--voxel-size 必须大于 0")

    return args




def create_pcd(record) :

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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 原来的 0.02 改为命令行参数。
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    return pcd





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

    selected_matches = matches[args.start_index::args.stride][:args.max_frames]

    results = []

    first_record = selected_matches[0]

    first_time = rgb_records[first_record["rgb_index"]][0]

    first_pcd = create_pcd(first_record)
    estimated_map = copy.deepcopy(first_pcd)
    reference_map = copy.deepcopy(first_pcd)

    results.append({
        "index": 0,
        "rotation_error_deg": 0.0,
        "translation_error_m": 0.0,
    })

    T_acc = np.eye(4)

    T_wc_0 = interpolate_pose(first_time, pose_timestamps, poses)

    for index, record in enumerate(selected_matches[:-1]):

        next_record = selected_matches[index + 1]

        source_time = rgb_records[next_record["rgb_index"]][0]

        target_pcd = create_pcd(record)  # 当前帧
        source_pcd = create_pcd(next_record)  # 下一帧

        T_step, history = icp_point_to_point(
            np.asarray(source_pcd.points),
            np.asarray(target_pcd.points),
            max_iterations=args.max_iterations,
            max_distance=args.max_distance,
            tolerance=args.tolerance,
        )

        T_acc = T_acc @ T_step

        source_pcd_copy = copy.deepcopy(source_pcd).transform(T_acc)

        estimated_map = estimated_map + source_pcd_copy

        T_wc_i = interpolate_pose(source_time, pose_timestamps, poses)

        T_ref_i = np.linalg.inv(T_wc_0) @ T_wc_i

        reference_points = copy.deepcopy(source_pcd).transform(T_ref_i)

        reference_map = reference_map + reference_points

        rotation_error_deg, translation_error_m = evaluate_pose(T_acc, T_ref_i)

        results.append({
            "index" : index+1,
            "rotation_error_deg" : rotation_error_deg,
            "translation_error_m" : translation_error_m
        })


    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "trajectory_errors.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


    if not o3d.io.write_point_cloud(str(args.output / "estimated_map.ply"), estimated_map):
        raise OSError("estimated_map 保存失败")

    if not o3d.io.write_point_cloud(str(args.output / "reference_map.ply"), reference_map):
        raise OSError("reference_map 保存失败")

    if args.show:
        o3d.visualization.draw_geometries([estimated_map], window_name="Estimated")
        o3d.visualization.draw_geometries([reference_map], window_name="Reference")





