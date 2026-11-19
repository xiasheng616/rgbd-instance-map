from pathlib import Path
import json
import numpy as np
import open3d as o3d
import argparse



def frame_time(frame_id):
    return float(frame_id.split("_")[1])

def voxel_keys(cloud, voxel_size=0.05):
    points = np.asarray(cloud.points)
    indices = np.floor(points / voxel_size).astype(np.int64)
    return {tuple(index) for index in indices}







parser = argparse.ArgumentParser()
parser.add_argument("--observations", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--min_voxel_iou", type=float, default=None)
args = parser.parse_args()

output_dir = args.output

clouds_dir = args.observations.parent

with (args.observations).open("r", encoding="utf-8") as f:
    observations = json.load(f)

frames = {}

for obs in observations["observations"] :
    frames.setdefault(obs["frame_id"], []).append(obs)

instances = {}
next_map_id = 1

for frame_id in sorted(frames, key=frame_time):

    available_ids = set(instances)

    for obs in frames[frame_id]:

        obs_cloud = o3d.io.read_point_cloud(str(clouds_dir / obs["cloud_file"]))

        obs_cloud_name = obs["cloud_file"]

        if obs_cloud.is_empty() :
            raise OSError(f"{obs_cloud_name} 读取失败")

        obs_cloud_voxel = voxel_keys(obs_cloud, voxel_size=0.05)

        obs_center = np.asarray(obs["centroid"])

        best_id = None
        best_distance = float("inf")

        for map_id in sorted(available_ids):

            instance = instances[map_id]

            if instance["label"] != obs["label"]:
                continue

            distance = np.linalg.norm(obs_center - instance["centroid"])

            instance_voxel = voxel_keys(instance["cloud"],voxel_size=0.05)

            IoU = len(obs_cloud_voxel & instance_voxel) / len(obs_cloud_voxel | instance_voxel)

            min_voxel_iou = args.min_voxel_iou  # 改成 None 时，只用质心距离

            passes_iou = min_voxel_iou is None or IoU >= min_voxel_iou

            if distance <= 0.5 and distance < best_distance and passes_iou:
                best_id = map_id
                best_distance = distance


        source = {"frame_id": frame_id, "local_id": obs["local_id"]}

        if best_id is None:
            instances[next_map_id] = {
                "label": obs["label"],
                "cloud": obs_cloud,
                "centroid": obs_center,
                "observations": [source],
            }
            print(f"{frame_id} 新建 map_id={next_map_id}")
            next_map_id += 1

            continue

        available_ids.remove(best_id)
        instance = instances[best_id]

        instance["cloud"] = (instance["cloud"] + obs_cloud).voxel_down_sample(0.02)

        merged_points = np.asarray(instance["cloud"].points)

        instance["centroid"] = merged_points.mean(axis=0)

        instance["observations"].append(source)

        print(f"{frame_id} 匹配 map_id={best_id}，距离={best_distance:.3f} 米")


results = []


for map_id, instance in instances.items() :

    output_dir.mkdir(parents=True, exist_ok=True)

    if not o3d.io.write_point_cloud(str(output_dir / f"instance_{map_id}.ply"), instance["cloud"]):
        raise OSError("点云保存失败")

    saved_points = np.asarray(instance["cloud"].points)

    centroid = saved_points.mean(axis=0)
    bbox_min = saved_points.min(axis=0)
    bbox_max = saved_points.max(axis=0)

    results.append({
        "map_id": map_id,
        "label": instance["label"],
        "point_count": saved_points.shape[0],
        "centroid": centroid.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "cloud_file": f"instance_{map_id}.ply",
        "observations": instance["observations"]
    })


instances_path = output_dir / "instances.json"
instances_path.parent.mkdir(parents=True, exist_ok=True)
with instances_path.open("w", encoding="utf-8") as f:
    json.dump({
        "scene_id": observations["scene_id"],
        "coordinate_frame": "world",
        "min_voxel_iou": args.min_voxel_iou,
        "match_distance_m": 0.5,
        "voxel_size_m": 0.02,
        "instances": results,
    }, f, ensure_ascii=False, indent=2)
