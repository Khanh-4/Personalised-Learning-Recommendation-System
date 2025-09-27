# src/evaluation.py
import logging
import numpy as np
from itertools import product

from models import train_hybrid
from evaluation_metrics import recall_at_k, ndcg_at_k   # ✅ import từ file riêng

def evaluate_models(models, test, user_col, item_col, k=10):
    results = {}
    users = test[user_col].unique()

    for name, model in models.items():
        all_recs, all_truths = [], []

        for u in users:
            recs = model.recommend(user_id=u, top_k=k)   # <-- thống nhất
            truths = test.loc[test[user_col] == u, item_col].tolist()

            all_recs.append(recs)
            all_truths.append(truths)

        recall = recall_at_k(all_recs, all_truths, k)
        ndcg = ndcg_at_k(all_recs, all_truths, k)

        results[name] = (recall, ndcg)
        print(f"{name}: Recall@{k}={recall:.3f}, NDCG@{k}={ndcg:.3f}")

    return results




def tune_hybrid_weights(models, test, user_col, item_col, step=0.1, k=5, verbose=False):
    """
    Grid search cho trọng số Hybrid theo Recall@k và NDCG@k.
    Nếu verbose=True: log tiến độ mỗi 10%.
    """
    weights_range = np.arange(0, 1.01, step)
    total = len(weights_range) ** 2
    logging.info(f"[Hybrid Tuning] Total configs = {total}")

    best_recall, best_ndcg = {"score": -1}, {"score": -1}
    tested = 0

    for w_pop in weights_range:
        for w_mf in weights_range:
            w_cb = 1.0 - w_pop - w_mf
            if w_cb < 0 or w_cb > 1:
                continue

            weights = {"pop": float(w_pop), "mf": float(w_mf), "cb": float(w_cb)}
            hybrid = train_hybrid(models, weights)

            metrics = evaluate_models({"Hybrid": hybrid}, test, user_col, item_col, k=k)
            recall, ndcg = metrics["Hybrid"]

            # update best
            if recall > best_recall["score"]:
                best_recall = {"pop": weights["pop"], "mf": weights["mf"], "cb": weights["cb"], "score": recall}
            if ndcg > best_ndcg["score"]:
                best_ndcg = {"pop": weights["pop"], "mf": weights["mf"], "cb": weights["cb"], "score": ndcg}

            tested += 1
            if verbose and tested % max(1, total // 10) == 0:  # chỉ log mỗi 10% nếu verbose
                logging.info(f"[Hybrid Tuning] Progress: {tested}/{total} tested")

    logging.info("\n=== BEST HYBRID CONFIG ===")
    logging.info(f"By Recall@{k}: pop={best_recall['pop']}, mf={best_recall['mf']}, cb={best_recall['cb']} → recall={best_recall['score']:.3f}")
    logging.info(f"By NDCG@{k}:  pop={best_ndcg['pop']}, mf={best_ndcg['mf']}, cb={best_ndcg['cb']} → ndcg={best_ndcg['score']:.3f}")

    return best_recall, best_ndcg


