import numpy as np
from scripts.evaluate_retrieval import evaluate_scores


feature_rows = [
    {"feature_index": 2, "map_id": 90},
    {"feature_index": 0, "map_id": 10},
    {"feature_index": 1, "map_id": 42},
]
scores = np.array([[0.90, 0.50, 0.10],
                   [0.60, 0.80, 0.20],
                   [0.80, 0.90, 0.70]])
queries = [
    {"query_id": f"q{i}", "text": f"synthetic query {i}", "target_map_id": target}
    for i, target in enumerate([10, 10, 90], start=1)
]


def main():
    results = evaluate_scores(scores, feature_rows, queries)
    ranks = [row["target_rank"] for row in results]
    top1_ids = [row["top1_map_id"] for row in results]

    assert ranks == [1, 2, 3], f"目标名次不对：{ranks}"
    assert top1_ids == [10, 42, 42], f"第一名的地图 ID 不对：{top1_ids}"
    np.testing.assert_allclose(
        [row["top1_cosine"] for row in results], [0.90, 0.80, 0.90]
    )

    # 用返回的名次独立核算指标，不加载模型或读取真实特征。
    top1_accuracy = sum(rank == 1 for rank in ranks) / len(ranks)
    mrr = sum(1 / rank for rank in ranks) / len(ranks)
    np.testing.assert_allclose([top1_accuracy, mrr], [1 / 3, 11 / 18])

    print(f"候选数：{scores.shape[1]}")
    print(f"目标名次：{ranks}")
    print(f"Top-1：{top1_accuracy:.6f} ({top1_accuracy:.2%})")
    print(f"MRR：{mrr:.6f}")
    print("合成验收通过")


if __name__ == "__main__":
    main()


