from copy import deepcopy
import open3d as o3d
from pathlib import Path
import json
import numpy as np






def make_pair(row, a_id, b_id) :

    cloud_a = o3d.io.read_point_cloud(str(source_dir / row["cloud_file"]))
    cloud_b = deepcopy(cloud_a)
    cloud_b.translate([0.40, 0.0, 0.0])

    obs_a = row.copy()
    obs_b = row.copy()
    obs_b["cloud_file"] = f'{row["frame_id"]}_b.ply'

    obs_a["local_id"] = a_id
    obs_b["local_id"] = b_id

    points_b = np.asarray(cloud_b.points)
    obs_b["centroid"] = points_b.mean(axis=0).tolist()
    obs_b["bbox_min"] = points_b.min(axis=0).tolist()
    obs_b["bbox_max"] = points_b.max(axis=0).tolist()


    output_dir.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_dir / obs_a["cloud_file"]), cloud_a):
        raise OSError("点云保存失败")
    if not o3d.io.write_point_cloud(str(output_dir / obs_b["cloud_file"]), cloud_b):
        raise OSError("点云保存失败")

    return obs_a, obs_b




source_dir = Path("outputs/stage03_observations")
output_dir = Path("outputs/stage03_observations_test")



with (source_dir / "observations.json").open("r", encoding="utf-8") as f:
    report = json.load(f)

first_row = report["observations"][0]
second_row = report["observations"][1]

a1, b1 = make_pair(first_row, a_id=1, b_id=2)
a2, b2 = make_pair(second_row, a_id=2, b_id=1)

test_report = report.copy()
test_report["observations"] = [a1, b2]


instances_path = output_dir / "observations.json"
instances_path.parent.mkdir(parents=True, exist_ok=True)
with instances_path.open("w", encoding="utf-8") as f:
    json.dump(test_report, f, ensure_ascii=False, indent=2)