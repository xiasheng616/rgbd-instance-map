from transformers import GemmaTokenizerFast, AutoModel
from pathlib import Path
import torch
import argparse
import numpy as np
import json


parser = argparse.ArgumentParser()
parser.add_argument("--features", type=Path, required=True)
parser.add_argument("--query", type=str, required=True)
parser.add_argument("--top-k", type=int, required=True)
args = parser.parse_args()

if args.top_k <= 0:
    parser.error("--top-k 必须大于 0")

query = args.query

features_path = args.features

with features_path.open("r", encoding="utf-8") as f:
    report = json.load(f)

rows =report["rows"]

instance_embeddings = np.load(features_path.parent / "instance_embeddings.npy")

model_dir = Path("outputs/stage04_models/siglip2-base-patch16-256")

device = "cuda" if torch.cuda.is_available() else "cpu"


model = AutoModel.from_pretrained(
    model_dir, use_safetensors=True, local_files_only=True
).to(device).eval()

tokenizer = GemmaTokenizerFast.from_pretrained(model_dir, local_files_only=True)
tokens = tokenizer(
    [query.lower()],
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt",
).to(device)

with torch.inference_mode() :
    encoded = model.get_text_features(input_ids=tokens["input_ids"])
    query_vector = torch.nn.functional.normalize(encoded.float(), dim=-1)[0].cpu().numpy()


scores = instance_embeddings @ query_vector
order = np.argsort(-scores)[:args.top_k]


rows_by_index = {row["feature_index"]: row for row in rows}

for rank, index in enumerate(order, start=1):
    row = rows_by_index[int(index)]
    print(
        f"rank={rank} map_id={row['map_id']} "
        f"cosine={scores[index]:.6f} "
        f"views={row['view_feature_indices']}"
    )

