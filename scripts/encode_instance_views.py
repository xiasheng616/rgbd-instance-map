from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import argparse
import json
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--views", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

output_dir = args.output

views_path = args.views

with views_path.open("r", encoding="utf-8") as f:
    views_report = json.load(f)

instances = views_report["instances"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = Path("outputs/stage04_models/siglip2-base-patch16-256")

processor = AutoImageProcessor.from_pretrained(
    model_dir, use_fast=False, local_files_only=True
)
model = AutoModel.from_pretrained(
    model_dir, use_safetensors=True, local_files_only=True
).to(device).eval()

images = []
rows = []

for instance in instances:
    for view in instance["views"]:
        crop_path = views_path.parent / view["crop_file"]
        with Image.open(crop_path) as image:
            images.append(image.convert("RGB"))

        rows.append({
            "feature_index": len(rows),
            "map_id": instance["map_id"],
            "frame_id": view["frame_id"],
            "local_id": view["local_id"],
            "crop_file": view["crop_file"],
        })


inputs = processor(images=images, return_tensors="pt").to(device)

with torch.inference_mode():
    features = model.get_image_features(**inputs)
    features = torch.nn.functional.normalize(features.float(), dim=-1)



output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "view_embeddings.npy", features.cpu().numpy())

with (output_dir / "features.json").open("w", encoding="utf-8") as f:
    json.dump({
        "scene_id": views_report["scene_id"],
        "model_id": "google/siglip2-base-patch16-256",
        "crop_mode": views_report["crop_mode"],
        "rows": rows,
    }, f, ensure_ascii=False, indent=2)


