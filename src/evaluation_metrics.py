# src/evaluation_metrics.py
# ============================================================
# Standard evaluation metrics for recommenders
# ============================================================

import numpy as np

def recall_at_k(all_recs, all_truths, k=10):
    hits, total = 0, 0
    for recs, truths in zip(all_recs, all_truths):
        if not truths:
            continue
        hits += len(set(recs[:k]) & set(truths))
        total += len(truths)
    return hits / total if total > 0 else 0.0


def ndcg_at_k(all_recs, all_truths, k=10):
    def dcg(scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))

    total_ndcg, count = 0, 0
    for recs, truths in zip(all_recs, all_truths):
        if not truths:
            continue
        scores = [1 if item in truths else 0 for item in recs[:k]]
        ideal_scores = sorted(scores, reverse=True)
        dcg_val = dcg(scores)
        idcg_val = dcg(ideal_scores)
        total_ndcg += dcg_val / idcg_val if idcg_val > 0 else 0
        count += 1
    return total_ndcg / count if count > 0 else 0.0
