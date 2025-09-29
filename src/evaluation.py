# src/evaluation.py
# ============================================================
# Evaluation utilities for recommenders
# ============================================================

import numpy as np
import logging
from models import train_hybrid, train_hybrid_ncf, MFRecommender
from evaluation_metrics import recall_at_k, ndcg_at_k

__all__ = [
    "evaluate_model",
    "evaluate_models",
    "tune_mf",
    "tune_hybrid_weights",
    "_safe_recommend"
]


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ------------------------------------------------------------
# Evaluate a single model
# ------------------------------------------------------------
def evaluate_model(model, test, user_col="learner_id", item_col="content_type", k=10):
    users = test[user_col].unique()
    all_recs, all_truths = [], []

    for u in users:
        recs = model.recommend(user_id=u, top_k=k)
        truths = test.loc[test[user_col] == u, item_col].tolist()
        all_recs.append(recs)
        all_truths.append(truths)

    recall = recall_at_k(all_recs, all_truths, k)
    ndcg = ndcg_at_k(all_recs, all_truths, k)
    return recall, ndcg


# ------------------------------------------------------------
# Evaluate multiple models at once
# (Modified to return both metrics + raw_scores for meta-hybrid)
# ------------------------------------------------------------
def evaluate_models(models, test, user_col="learner_id", item_col="content_type", k=10):
    metrics = {}
    raw_scores = {}
    users = test[user_col].unique()

    for name, model in models.items():
        all_recs, all_truths = [], []
        all_scores = {}

        for u in users:
            # try để tránh khác biệt tham số recommend
            try:
                recs = model.recommend(user_id=u, top_k=k)
            except TypeError:
                recs = model.recommend(u, k)

            truths = test.loc[test[user_col] == u, item_col].tolist()
            all_recs.append(recs)
            all_truths.append(truths)

            # raw_scores: rank-based scoring
            score_dict = {}
            for idx, i in enumerate(recs):
                score_dict[i] = 1.0 / (idx + 1)  # rank → score
            all_scores[u] = score_dict

        recall = recall_at_k(all_recs, all_truths, k)
        ndcg = ndcg_at_k(all_recs, all_truths, k)

        metrics[name] = (recall, ndcg)
        raw_scores[name] = all_scores

        print(f"{name}: Recall@{k}={recall:.3f}, NDCG@{k}={ndcg:.3f}")

    return metrics, raw_scores


# ------------------------------------------------------------
# Tune MF parameters
# ------------------------------------------------------------
def tune_mf(train, val, user_col="learner_id", item_col="content_type", rating_col="engagement_score"):
    configs = [
        {"factors": 20, "lr": 0.01, "reg": 0.01, "epochs": 20},
        {"factors": 50, "lr": 0.01, "reg": 0.01, "epochs": 30},
        {"factors": 100, "lr": 0.005, "reg": 0.01, "epochs": 50},
        {"factors": 50, "lr": 0.005, "reg": 0.05, "epochs": 40},
    ]
    best_cfg, best_score = None, -1

    for cfg in configs:
        print(f"\n[MF Tuning] Trying config {cfg}")
        mf = MFRecommender(
            num_factors=cfg["factors"],
            learning_rate=cfg["lr"],
            reg=cfg["reg"],
            epochs=cfg["epochs"]
        )
        mf.fit(train, user_col=user_col, item_col=item_col, rating_col=rating_col)

        recall, ndcg = evaluate_model(mf, val, user_col, item_col, k=10)
        print(f"[MF Tuning] Recall@10={recall:.3f}, NDCG@10={ndcg:.3f}")

        if recall > best_score:
            best_score = recall
            best_cfg = cfg

    print("\n=== Best MF Config ===")
    print(best_cfg, "Recall@10:", best_score)
    return best_cfg


# ------------------------------------------------------------
# Tune Hybrid Weights (MF / NCF)
# ------------------------------------------------------------
def tune_hybrid_weights(models, test, user_col, item_col, step=0.1, k=10, verbose=False):
    """
    Grid search for Hybrid weights based on Recall@k and NDCG@k.
    Supports both MF-Hybrid and NCF-Hybrid.
    """
    keys = list(models.keys())
    use_ncf = "ncf" in keys

    weights_range = np.arange(0, 1.01, step)
    total = len(weights_range) ** 2
    logging.info(f"[Hybrid Tuning] Total configs = {total}")

    best_recall, best_ndcg = {"score": -1}, {"score": -1}
    tested = 0

    for w_pop in weights_range:
        for w_mid in weights_range:  # mf or ncf
            w_cb = 1.0 - w_pop - w_mid
            if w_cb < 0 or w_cb > 1:
                continue

            if use_ncf:
                weights = {"pop": float(w_pop), "ncf": float(w_mid), "cb": float(w_cb)}
                hybrid = train_hybrid_ncf(models, weights)
            else:
                weights = {"pop": float(w_pop), "mf": float(w_mid), "cb": float(w_cb)}
                hybrid = train_hybrid(models, weights)

            # chỉ lấy metrics, bỏ raw_scores
            metrics, _ = evaluate_models({"Hybrid": hybrid}, test, user_col, item_col, k=k)
            recall, ndcg = metrics["Hybrid"]

            if recall > best_recall["score"]:
                best_recall = {**weights, "score": recall}
            if ndcg > best_ndcg["score"]:
                best_ndcg = {**weights, "score": ndcg}

            tested += 1
            if verbose and tested % max(1, total // 10) == 0:
                print(f"[Hybrid Tuning] Progress {tested}/{total}...")

    print("\n=== BEST HYBRID CONFIG ===")
    if use_ncf:
        print(f"By Recall@{k}: pop={best_recall['pop']}, ncf={best_recall['ncf']}, cb={best_recall['cb']} → recall={best_recall['score']:.3f}")
        print(f"By NDCG@{k}:  pop={best_ndcg['pop']}, ncf={best_ndcg['ncf']}, cb={best_ndcg['cb']} → ndcg={best_ndcg['score']:.3f}")
    else:
        print(f"By Recall@{k}: pop={best_recall['pop']}, mf={best_recall['mf']}, cb={best_recall['cb']} → recall={best_recall['score']:.3f}")
        print(f"By NDCG@{k}:  pop={best_ndcg['pop']}, mf={best_ndcg['mf']}, cb={best_ndcg['cb']} → ndcg={best_ndcg['score']:.3f}")

    return best_recall, best_ndcg

# ------------------------------------------------------------
# Safe recommend wrapper (for models with different signatures)
# ------------------------------------------------------------
def _safe_recommend(model, user, top_k=10):
    """
    Try different calling conventions for recommend() 
    because some models use (user_id=..., top_k=...), 
    some use (user, k), etc.
    """
    try:
        return model.recommend(user_id=user, top_k=top_k)
    except TypeError:
        try:
            return model.recommend(user, top_k=top_k)
        except TypeError:
            try:
                return model.recommend(user, top_k)
            except TypeError:
                try:
                    return model.recommend(user)
                except Exception:
                    return []
