from rgbd_map.registration import icp_point_to_point,evaluate_alignment,evaluate_pose
from rgbd_map.tum_io import read_image_index,read_pose_index,associate_records
from rgbd_map.geometry import interpolate_pose
from pathlib import Path
import cv2 as cv
import numpy as np
import open3d as o3d
import copy
import argparse
import csv


def show_registration(source_pcd, target_pcd, T, title):
    # 每次使用副本，保护原始点云
    source_view = copy.deepcopy(source_pcd)
    target_view = copy.deepcopy(target_pcd)

    source_view.paint_uniform_color([1.0, 0.5, 0.0])  # 源：橙色
    target_view.paint_uniform_color([0.0, 0.6, 1.0])  # 目标：蓝色

    # 只移动源点云，目标保持不动
    source_view.transform(T)

    o3d.visualization.draw_geometries(
        [source_view, target_view],
        window_name=title,
        front=[0, 0, -1],
        up=[0, -1, 0],
        lookat=target_view.get_center(),
        zoom=0.7,
    )


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


def parse_args():
    parser = argparse.ArgumentParser(description="RGB-D 两帧 ICP 配准与评价")
    parser.add_argument("--source", "--source-index", type=int, default=10)
    parser.add_argument("--target", "--target-index", type=int, default=0)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-distance", type=float, default=0.10)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.source < 0 or args.target < 0:
        parser.error("--source 和 --target 必须大于等于 0")
    if args.max_iterations <= 0:
        parser.error("--max-iterations 必须大于 0")
    if not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("--voxel-size 必须是大于 0 的有限数")
    if not np.isfinite(args.max_distance) or args.max_distance <= 0:
        parser.error("--max-distance 必须是大于 0 的有限数")
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance 必须是大于 0 的有限数")

    return args


if __name__ == "__main__":

    args = parse_args()
    sequence_dir = args.sequence

    rgb_records = read_image_index(sequence_dir / "rgb.txt")
    depth_records = read_image_index(sequence_dir / "depth.txt")
    pose_timestamps ,poses = read_pose_index(sequence_dir / "groundtruth.txt")

    rgb_path = sequence_dir / rgb_records[0][1]
    depth_path = sequence_dir / depth_records[0][1]

    matches = associate_records(
        rgb_records,
        depth_records,
        pose_timestamps,
    )

    if args.source >= len(matches) or args.target >= len(matches):
        raise ValueError(f"帧索引越界：共有 {len(matches)} 组关联记录")

    target_record = matches[args.target]
    source_record = matches[args.source]

    source_time = rgb_records[source_record["rgb_index"]][0]
    target_time = rgb_records[target_record["rgb_index"]][0]

    target_pcd = create_pcd(target_record)
    source_pcd = create_pcd(source_record)

    T_own, history = icp_point_to_point(
        np.asarray(source_pcd.points),
        np.asarray(target_pcd.points),
        max_iterations=args.max_iterations,
        max_distance=args.max_distance,
        tolerance=args.tolerance,
    )

    reference = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        args.max_distance,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1.0,
            relative_rmse=args.tolerance,
            max_iteration=args.max_iterations,
        ),
    )

    states = [
        ("初始状态", np.eye(4)),
        ("自写 ICP", T_own),
        ("Open3D ICP", reference.transformation),
    ]

    # 新增：记录实际迭代次数和停止原因。
    if len(history) >= 2 and abs(history[-1] - history[-2]) <= args.tolerance:
        stop_reason = "rmse_tolerance"
    else:
        stop_reason = "max_iterations"
    print("自写 ICP 迭代次数：", len(history), "停止原因：", stop_reason)

    # 新增：将两个评价循环的结果按名称收集，最后写入 CSV。
    results = {}
    for name, T in states:
        num_matches, fitness, inlier_rmse = evaluate_alignment(
            source_pcd.points,
            target_pcd.points,
            T,
            max_distance=args.max_distance,
        )

        print(
            f"{name}：有效对应数={num_matches}，"
            f"匹配比例={fitness:.2%}，"
            f"RMSE={inlier_rmse * 1000:.3f} 毫米"
        )

        results[name] = {
            "method": name,
            "sequence": str(sequence_dir.resolve()),
            "source_index": args.source,
            "target_index": args.target,
            "voxel_size_m": args.voxel_size,
            "max_distance_m": args.max_distance,
            "max_iterations": args.max_iterations,
            "rmse_tolerance_m": args.tolerance,
            "num_matches": num_matches,
            "fitness": fitness,
            "inlier_rmse_m": inlier_rmse,
            "num_updates": len(history) if name == "自写 ICP" else "",
            "stop_reason": stop_reason if name == "自写 ICP" else "",
        }

    T_wc_source = interpolate_pose(source_time,pose_timestamps,poses)
    T_wc_target = interpolate_pose(target_time,pose_timestamps,poses)

    T_ref = np.linalg.inv(T_wc_target) @ T_wc_source

    for name, T in states:
        rotation_error_deg,translation_error_m = evaluate_pose(
            T,
            T_ref
        )

        print(
            f"{name}：旋转角误差={rotation_error_deg}度，"
            f"平移误差={translation_error_m}米"
        )

        results[name]["rotation_error_deg"] = rotation_error_deg
        results[name]["translation_error_m"] = translation_error_m

    # 新增：统一保存到 --output 指定的目录。
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results["初始状态"]))
        writer.writeheader()
        writer.writerows(results.values())

    np.save(args.output / "T_own.npy", T_own)
    np.save(args.output / "T_open3d.npy", reference.transformation)
    np.save(args.output / "T_ref.npy", T_ref)

    with (args.output / "icp_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "pre_update_rmse_m"])
        writer.writerows(enumerate(history))

    if not o3d.io.write_point_cloud(str(args.output / "target.ply"), target_pcd):
        raise OSError("target 点云保存失败")
    for filename, T in [
        ("aligned_own_icp.ply", T_own),
        ("aligned_open3d_icp.ply", reference.transformation),
    ]:
        aligned_pcd = copy.deepcopy(source_pcd)
        aligned_pcd.transform(T)
        if not o3d.io.write_point_cloud(str(args.output / filename), aligned_pcd):
            raise OSError(f"点云保存失败：{filename}")

    print("结果已保存：", args.output)

    # 原来的三次显示，改为由 --show 控制。
    if args.show:
        show_registration(
            source_pcd, target_pcd,
            np.eye(4),
            "1 - Initial",
        )
        show_registration(
            source_pcd, target_pcd,
            T_own,
            "2 - Own ICP",
        )
        show_registration(
            source_pcd, target_pcd,
            reference.transformation,
            "3 - Open3D ICP",
        )
