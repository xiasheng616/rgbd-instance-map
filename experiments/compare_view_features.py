from pathlib import Path
import json
import numpy as np


def load_vectors(folder):
    matrix = np.load(folder / "view_embeddings.npy")
    with (folder / "features.json").open("r", encoding="utf-8") as f:
        report = json.load(f)

    rows = report["rows"]

    by_view = {}
    for row in rows:
        key = (row["map_id"], row["frame_id"], row["local_id"])
        by_view[key] = matrix[row["feature_index"]]

    return by_view


box_vectors = load_vectors(Path("outputs/stage04_features_box"))
mask_vectors = load_vectors(Path("outputs/stage04_features_mask"))


if box_vectors.keys() != mask_vectors.keys() :
    raise ValueError(f"mask的key {mask_vectors.keys()} 与box的key {box_vectors.keys()} 不一致")

for key in box_vectors.keys() :
    score = float(np.dot(box_vectors[key], mask_vectors[key]))
    print(f"{key[1]}  cosine={score:.6f}")