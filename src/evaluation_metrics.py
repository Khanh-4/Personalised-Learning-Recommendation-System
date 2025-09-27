import numpy as np

def recall_at_k(all_recs, all_truths, k=5, item_col=None):
    """
    Tính Recall@k trung bình cho tất cả users.
    all_recs  : list of list, mỗi phần tử là top-k recommendation cho 1 user
    all_truths: list of list, mỗi phần tử là ground truth items cho 1 user
    """
    recalls = []
    for recs, truths in zip(all_recs, all_truths):
        if not truths:  # user không có ground-truth trong test
            continue
        hits = len(set(recs[:k]) & set(truths))
        recalls.append(hits / len(set(truths)))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(all_recs, all_truths, k=5, item_col=None):
    """
    Tính NDCG@k trung bình cho tất cả users.
    """
    ndcgs = []
    for recs, truths in zip(all_recs, all_truths):
        if not truths:
            continue

        dcg = 0.0
        for idx, item in enumerate(recs[:k]):
            if item in truths:
                dcg += 1.0 / np.log2(idx + 2)  # idx+2 vì log2(1+rank)

        # ideal DCG
        ideal_hits = min(len(truths), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0
