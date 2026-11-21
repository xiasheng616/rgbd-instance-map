from pathlib import Path
import json
from shutil import copy2


output_dir = Path("outputs/stage04_retrieval_inputs")


chair_observations_path = Path("outputs/stage03_observations/observations.json")
bed_observations_path = Path("outputs/stage03_bed_observations/observations.json")

chair_mask_path = Path("outputs/stage03_chair_views_masks/results.json")
bed_mask_path = Path("outputs/stage03_bed_masks/results.json")

with chair_observations_path.open("r", encoding="utf-8") as f:
    chair_observations = json.load(f)

with bed_observations_path.open("r", encoding="utf-8") as f:
    bed_observations = json.load(f)

with chair_mask_path.open("r", encoding="utf-8") as f:
    chair_mask = json.load(f)

with bed_mask_path.open("r", encoding="utf-8") as f:
    bed_mask = json.load(f)


output_dir.mkdir(parents=True, exist_ok=True)

for chair_obs in chair_observations["observations"] :
    copy2(chair_observations_path.parent / chair_obs["cloud_file"], output_dir / chair_obs["cloud_file"])

for bed_obs in bed_observations["observations"] :
    copy2(bed_observations_path.parent / bed_obs["cloud_file"], output_dir / bed_obs["cloud_file"])

chair_observations["observations"].extend(bed_observations["observations"])

with (output_dir / "observations.json").open("w", encoding="utf-8") as f:
    json.dump(chair_observations, f, ensure_ascii=False, indent=2)

chair_mask["images"].extend(bed_mask["images"])

with (output_dir / "detections.json").open("w", encoding="utf-8") as f:
    json.dump(chair_mask, f, ensure_ascii=False, indent=2)