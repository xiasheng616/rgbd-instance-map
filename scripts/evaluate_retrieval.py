from transformers import GemmaTokenizerFast, AutoModel
from pathlib import Path
import torch
import argparse
import numpy as np
import json


def evaluate_scores(scores, feature_rows, queries) :

    results = []

    rows_by_index = {feature_row["feature_index"]: feature_row for feature_row in feature_rows}

    for score_row, query in zip(scores, queries):
        order = np.argsort(-score_row, kind="stable")
        ranked_ids = [rows_by_index[int(i)]["map_id"] for i in order]

        top1_map_id = ranked_ids[0]
        top1_cosine = float(score_row[order[0]])
        
        if "target_map_ids" in query:
            target_ids = set(query["target_map_ids"])
        else:
            target_ids = {query["target_map_id"]}

        if not target_ids or not target_ids.issubset(ranked_ids):
            raise ValueError(f"{query['query_id']} 的正确 ID 为空或不在候选库中")

        target_rank = next(
            rank for rank, map_id in enumerate(ranked_ids, start=1)
            if map_id in target_ids
        )

        results.append({
            **query,
            "top1_map_id": int(top1_map_id),
            "top1_cosine": top1_cosine,
            "target_rank": target_rank,
        })

    return results




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output

    queries_path = args.queries

    features_path = args.features

    with queries_path.open("r", encoding="utf-8") as f:
         queries_report = json.load(f)

    with features_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    if queries_report["scene_id"] != report["scene_id"]:
        parser.error("查询清单和实例特征库不是同一个场景")

    query_rows = queries_report["queries"]
    query_texts = [row["text"] for row in query_rows]

    rows =report["rows"]

    instance_embeddings = np.load(features_path.parent / "instance_embeddings.npy")

    model_dir = Path("outputs/stage04_models/siglip2-base-patch16-256")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = AutoModel.from_pretrained(
        model_dir, use_safetensors=True, local_files_only=True
    ).to(device).eval()

    tokenizer = GemmaTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    tokens = tokenizer(
        [text.lower() for text in query_texts],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode() :
        encoded = model.get_text_features(input_ids=tokens["input_ids"])
        query_matrix = torch.nn.functional.normalize(encoded.float(), dim=-1).cpu().numpy()


    scores = query_matrix @ instance_embeddings.T  # (Q, N)

    # tokenizer 使用 query_texts；算出 scores 后：
    results = evaluate_scores(scores, rows, query_rows)

    ranks = [result["target_rank"] for result in results]

    num_queries = len(ranks)

    if num_queries == 0:
        raise ValueError("没有查询结果，无法计算指标")

    # Top-1：正确目标排第一的查询数 ÷ 查询总数
    num_correct = sum(1 for rank in ranks if rank == 1)
    top1_accuracy = num_correct / num_queries

    # MRR：每条查询的正确名次取倒数，再求平均
    mrr = sum(1 / rank for rank in ranks) / num_queries

    summary = {
        "scene_id": report["scene_id"],
        "num_queries": num_queries,
        "num_candidates": int(instance_embeddings.shape[0]),
        "top1_accuracy": top1_accuracy,
        "mrr": mrr,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "per_query.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)





