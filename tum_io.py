from pathlib import Path
import cv2 as cv
import numpy as np
import open3d as o3d

def read_image_index(path):
    """返回 [(时间戳, 图片相对路径), ...]，按时间排序。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f :
            line=line.strip()
            if not line or line.startswith("#"):
                continue

            timestamp, image_path=line.split()

            timestamp=float(timestamp)

            records.append((timestamp, image_path))

    records.sort(key=lambda x:x[0])

    return records




def read_pose_index(path):
    """返回：
    timestamps：形状 (N,)
    poses：形状 (N, 7)
    每个位姿为 tx, ty, tz, qx, qy, qz, qw
    """
    timestamps=[]
    poses=[]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            timestamp,tx,ty,tz,qx,qy,qz,qw = line.split()

            timestamps.append(float(timestamp))
            poses.append([float(x) for x in (tx,ty,tz,qx,qy,qz,qw)])

    timestamps = np.asarray(timestamps, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)

    order = np.argsort(timestamps)
    return timestamps[order], poses[order]



def associate_records(
    rgb_records,
    depth_records,
    pose_timestamps,
    max_diff=0.02,
):
    """返回关联成功的记录列表。"""
    associate_records=[]

    used_depth = set()
    for rgb_index, (rgb_timestamp, rgb_path) in enumerate(rgb_records):
        best_depth_index = None
        best_diff = float("inf")  # 正无穷，方便后面不断更新最小值
        best_pose_index = None

        for depth_index, (depth_timestamp, depth_path) in enumerate(depth_records):
            if depth_index in used_depth:
                continue  # 已使用，跳过这张深度图

            diff_depth = np.abs(depth_timestamp - rgb_timestamp)

            if diff_depth < best_diff :
                best_diff=diff_depth
                best_depth_index = depth_index

        if best_depth_index is None or best_diff > max_diff:
            continue

        diff_pose = np.abs(pose_timestamps - rgb_timestamp)
        best_pose_index = np.argmin(diff_pose)

        if diff_pose[best_pose_index] < max_diff :

            used_depth.add(best_depth_index)
            associate_records.append({
                "rgb_index": rgb_index,
                "depth_index": best_depth_index,
                "pose_index": best_pose_index,
                "rgb_depth_dt": best_diff,
                "rgb_pose_dt": diff_pose[best_pose_index]
            })

        else :
            continue

    return associate_records







if __name__ == "__main__":

    sequence_dir = Path(
        r"E:\shixi\rgbd_instance_map\data\tum\rgbd_dataset_freiburg1_desk"
    )

    rgb_records = read_image_index(sequence_dir / "rgb.txt")
    depth_records = read_image_index(sequence_dir / "depth.txt")
    pose_timestamps,poses = read_pose_index(sequence_dir / "groundtruth.txt")

    print("RGB 数量：", len(rgb_records))
    print("RGB 前两条：", rgb_records[:2])

    print("深度数量：", len(depth_records))
    print("深度前两条：", depth_records[:2])

    print("位姿数量：", len(poses))
    print("位姿前两条：", poses[:2])

    rgb_path = sequence_dir / rgb_records[0][1]
    depth_path = sequence_dir / depth_records[0][1]

    image=cv.imread(str(rgb_path))
    print("第一张 RGB 的 shape:",image.shape," dtype:",image.dtype)

    depth=cv.imread(str(depth_path),cv.IMREAD_UNCHANGED)
    print("第一张深度的 shape:",depth.shape," dtype:",depth.dtype)

    print("时间戳：", pose_timestamps.shape, pose_timestamps.dtype)
    print("位姿：", poses.shape, poses.dtype)

    matches = associate_records(
        rgb_records,
        depth_records,
        pose_timestamps,
    )

    print("关联成功数量：", len(matches))
    print("未关联 RGB 数量：", len(rgb_records) - len(matches))
    print("前 5 条关联：", matches[:5])

    depth_indices = [record["depth_index"] for record in matches]
    print("深度是否重复：", len(depth_indices) != len(set(depth_indices)))

    if matches:
        for key in ("rgb_depth_dt", "rgb_pose_dt"):
            differences = [record[key] for record in matches]
            print(key, "最大值：", max(differences),
                  "平均值：", np.mean(differences))







