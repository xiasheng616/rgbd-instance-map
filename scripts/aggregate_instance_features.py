from pathlib import Path
import argparse
import json
import numpy as np





parser = argparse.ArgumentParser()
parser.add_argument("--features", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

output_dir = args.output
features_path = args.features

with features_path.open("r", encoding="utf-8") as f:
    report = json.load(f)

by_instance = {}
for row in report["rows"]:
    by_instance.setdefault(row["map_id"], []).append(row["feature_index"])

matrix = np.load(features_path.parent / "view_embeddings.npy")

vectors = []
instance_rows = []


for map_id in sorted(by_instance):
    indices = by_instance[map_id]
    vector = matrix[indices].mean(axis=0)

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"map_id={map_id} 的平均向量范数为 0")

    instance_rows.append({
        "feature_index": len(vectors),
        "map_id": map_id,
        "view_count": len(indices),
        "view_feature_indices": indices,
    })
    vectors.append(vector / norm)


instance_embeddings = np.stack(vectors, axis=0)

output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "instance_embeddings.npy", instance_embeddings)

with (output_dir / "instance_features.json").open("w", encoding="utf-8") as f:
    json.dump({
        "scene_id": report["scene_id"],
        "model_id": report["model_id"],
        "crop_mode": report["crop_mode"],
        "rows": instance_rows,
    }, f, ensure_ascii=False, indent=2)
