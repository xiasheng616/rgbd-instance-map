import json
from pathlib import Path
from shutil import copy2

source = Path("outputs/stage04_features_box")
test_dir = Path("outputs/stage04_features_aggregation_test")
test_dir.mkdir(parents=True, exist_ok=True)

with (source / "features.json").open(encoding="utf-8") as f:
    report = json.load(f)

rows = report["rows"]
report["rows"] = [
    dict(rows[2], map_id=42),
    dict(rows[0], map_id=10),
    dict(rows[1], map_id=10),
]

copy2(source / "view_embeddings.npy", test_dir / "view_embeddings.npy")
with (test_dir / "features.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)