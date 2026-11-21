from pathlib import Path
import json
from math import ceil, floor
import cv2 as cv
import argparse
import numpy as np




parser = argparse.ArgumentParser()
parser.add_argument("--instances", type=Path, required=True)
parser.add_argument("--detections", type=Path, required=True)
parser.add_argument("--crop-mode", choices=("box", "mask"), default="box")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

output_dir = args.output
instances_path = args.instances
detection_path = args.detections

with detection_path.open("r", encoding="utf-8") as f:
    results = json.load(f)

with instances_path.open("r", encoding="utf-8") as f:
    map_report = json.load(f)


instances = map_report["instances"]

lookup = {}

# 先把所有帧的检测放进 lookup
for image_info in results["images"]:
    frame_id = Path(image_info["image"]).stem

    for detection in image_info["detections"]:
        mask_path = detection_path.parent / detection["mask_file"]
        key = (frame_id, detection["object_id"])
        lookup[key] = (image_info["image"], detection)

output_dir.mkdir(parents=True, exist_ok=True)
exported_instances = []

for instance in instances:

    map_id = instance["map_id"]
    views = []

    for obs in instance["observations"]:

        frame_id = obs["frame_id"]
        local_id = obs["local_id"]

        image_path, detection = lookup[(frame_id, local_id)]

        image = cv.imread(image_path)

        if image is None:
            raise OSError(f"读取失败：{image_path}")

        if args.crop_mode == "box" :

            x1, y1, x2, y2 = detection["box_xyxy"]
            height, width = image.shape[:2]
            crop = image[max(0, floor(y1)):min(height, ceil(y2)),
            max(0, floor(x1)):min(width, ceil(x2))]


        elif args.crop_mode == "mask" :

            mask_path = detection_path.parent / detection["mask_file"]
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

            if mask is None:
                raise OSError(f"掩码读取失败：{mask_path}")

            if mask.shape != image.shape[:2]:
                raise ValueError(f"掩码尺寸 {mask.shape} 与原图尺寸 {image.shape[:2]} 不一致：{mask_path}")

            ys, xs = np.nonzero(mask > 0)
            if xs.size == 0:
                raise ValueError(f"掩码没有前景：{mask_path}")

            crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]



        crop_file = f"instance_{map_id}_{frame_id}_{local_id}.png"

        if crop.size == 0 or not cv.imwrite(str(output_dir / crop_file), crop):
            raise OSError(f"裁剪图保存失败：{crop_file}")

        views.append({
            "frame_id": frame_id,
            "local_id": local_id,
            "crop_file": crop_file,
        })

    exported_instances.append({
        "map_id": map_id,
        "label": instance["label"],
        "views": views,
    })


with (output_dir / "views.json").open("w", encoding="utf-8") as f:
    json.dump({
        "scene_id": map_report["scene_id"],
        "crop_mode": args.crop_mode,
        "instances": exported_instances,
    }, f, ensure_ascii=False, indent=2)
