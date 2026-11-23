from transformers import GemmaTokenizerFast, AutoModel
from pathlib import Path
import torch
import argparse
import numpy as np
import json
from shutil import copy2





def select_first_views(rows):
    by_map = {}
    for row in rows:
        by_map.setdefault(row["map_id"], []).append(row)

    selected = []
    for map_id in sorted(by_map):
        first = min(
            by_map[map_id],
            key=lambda row: (
                float(row["frame_id"].rsplit("_", 1)[1]),
                row["feature_index"],
            ),
        )
        selected.append(first)

    return selected



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output

    features_path = args.features

    with features_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    rows = report["rows"]

    selected_rows = select_first_views(rows)

    first_view_report = report.copy()
    first_view_report["rows"] = selected_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "features.json").open("w", encoding="utf-8") as f:
        json.dump(first_view_report, f, ensure_ascii=False, indent=2)

    copy2(features_path.parent / "view_embeddings.npy", output_dir / "view_embeddings.npy")