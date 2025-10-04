# src/evaluation.py
# ============================================================
# Evaluation utilities for recommenders
# ============================================================

import numpy as np
import logging
from models import train_hybrid, train_hybrid_ncf, MFRecommender
from evaluation_metrics import recall_at_k, ndcg_at_k
from tabulate import tabulate
import pandas as pd
import os

__all__ = [
    "evaluate_model",
    "evaluate_models",
    "tune_mf",
    "tune_hybrid_weights",
    "train_hybrid_generic",
    "_safe_recommend",
    "evaluate_and_log"
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
        recs = _safe_recommend(model, u, top_k=k)
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
            recs = _safe_recommend(model, u, top_k=k)
            truths = test.loc[test[user_col] == u, item_col].tolist()
            all_recs.append(recs)
            all_truths.append(truths)

            # raw_scores: rank-based scoring
            score_dict = {}
            if isinstance(recs, list):
                for idx, i in enumerate(recs):
                    if isinstance(i, tuple):
                        item, sc = i
                        score_dict[item] = sc
                    else:
                        score_dict[i] = 1.0 / (idx + 1)
            elif isinstance(recs, dict):
                score_dict = recs
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
# Generalized Hybrid Weight Tuning
# ------------------------------------------------------------
def tune_hybrid_weights(models, test, user_col, item_col, step=0.1, k=10, verbose=False):
    import itertools
    keys = list(models.keys())
    n = len(keys)

    weights_range = np.arange(0, 1.01, step)
    total = len(weights_range) ** n
    logging.info(f"[Hybrid Tuning] Total configs = {total}")

    best_recall, best_ndcg = {"score": -1}, {"score": -1}
    tested = 0

    for wp in itertools.product(weights_range, repeat=n):
        if abs(sum(wp) - 1.0) > 1e-6:
            continue
        weights = dict(zip(keys, map(float, wp)))

        hybrid = train_hybrid_generic(models, weights)
        recall, ndcg = evaluate_model(hybrid, test, user_col, item_col, k=k)

        if recall > best_recall["score"]:
            best_recall = {**weights, "score": recall}
        if ndcg > best_ndcg["score"]:
            best_ndcg = {**weights, "score": ndcg}

        tested += 1
        if verbose and tested % max(1, total // 10) == 0:
            print(f"[Hybrid Tuning] Progress {tested}/{total}...")

    print("\n=== BEST HYBRID CONFIG ===")
    print(f"By Recall@{k}: {best_recall} ")
    print(f"By NDCG@{k}:  {best_ndcg} ")

    return best_recall, best_ndcg


# ------------------------------------------------------------
# Generic Hybrid Recommender
# ------------------------------------------------------------
def train_hybrid_generic(models, weights):
    """
    Generic hybrid recommender that linearly combines scores 
    from multiple models using given weights.
    """
    class HybridRecommender:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights

        def recommend(self, user_id, top_k=10):
            scores = {}
            for name, model in self.models.items():
                w = self.weights.get(name, 0.0)
                if w <= 0:
                    continue
                try:
                    recs = _safe_recommend(model, user_id, top_k=top_k*2)
                except Exception:
                    continue

                # normalize về dict {item: score}
                rec_dict = {}
                if isinstance(recs, list):
                    for idx, i in enumerate(recs):
                        if isinstance(i, tuple):
                            item, sc = i
                            rec_dict[item] = sc * w
                        else:
                            rec_dict[i] = (1.0 / (idx + 1)) * w
                elif isinstance(recs, dict):
                    rec_dict = {i: s * w for i, s in recs.items()}

                for iid, sc in rec_dict.items():
                    scores[iid] = scores.get(iid, 0) + sc

            if not scores:
                return []
            return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    return HybridRecommender(models, weights)


# ------------------------------------------------------------
# Safe recommend wrapper (for models with different signatures)
# ------------------------------------------------------------
def _safe_recommend(model, user, top_k=10):
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
                else:
                    return model.recommend(user)[:top_k]
            else:
                return model.recommend(user, top_k)
        else:
            return model.recommend(user, top_k)
    else:
        return model.recommend(user_id=user, top_k=top_k)


# ------------------------------------------------------------
# Precision & MAP helpers
# ------------------------------------------------------------
def precision_at_k(recommended, relevant, k=10):
    if not recommended:
        return 0.0
    recommended_at_k = recommended[:k]
    hits = sum(1 for i in recommended_at_k if i in relevant)
    return hits / k

def average_precision(recommended, relevant, k=10):
    if not recommended:
        return 0.0
    score, hits = 0.0, 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


# ------------------------------------------------------------
# Evaluate & Log all models (with CSV/TXT/PDF export)
# ------------------------------------------------------------
def evaluate_and_log(models, test, user_col="learner_id", item_col="content_type", k=10, save_path=None):
    results = []
    users = test[user_col].unique()

    for name, model in models.items():
        all_recs, all_truths = [], []
        precs, maps = [], []


        for u in users:
            recs = _safe_recommend(model, u, top_k=k)
            truths = test.loc[test[user_col] == u, item_col].tolist()
            all_recs.append(recs)
            all_truths.append(truths)

            precs.append(precision_at_k(recs, truths, k))
            maps.append(average_precision(recs, truths, k))

        recall = recall_at_k(all_recs, all_truths, k)
        ndcg = ndcg_at_k(all_recs, all_truths, k)
        precision = np.mean(precs) if precs else 0.0
        map_score = np.mean(maps) if maps else 0.0

        results.append([name, round(recall,4), round(ndcg,4), round(precision,4), round(map_score,4)])

    # ✅ Print table
    print("\n=== 📊 Evaluation Summary ===")
    print(tabulate(results, headers=["Model", f"Recall@{k}", f"NDCG@{k}", f"Precision@{k}", f"MAP@{k}"], tablefmt="github"))

    # ✅ Save CSV/TXT/PDF
    if save_path:
        df = pd.DataFrame(results, columns=["Model", f"Recall@{k}", f"NDCG@{k}", f"Precision@{k}", f"MAP@{k}"])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Metrics saved to {save_path}")

        txt_path = save_path.replace(".csv", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(tabulate(results, headers=df.columns, tablefmt="github"))
        print(f"✅ Summary TXT saved: {txt_path}")

        try:
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet

            pdf_path = save_path.replace(".csv", ".pdf")
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Evaluation Summary", styles["Heading1"]))
            table = Table([df.columns.tolist()] + results)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("TEXTCOLOR", (0,0), (-1,0), colors.black),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ]))
            elements.append(table)
            doc.build(elements)
            print(f"📄 PDF report generated: {pdf_path}")
        except Exception as e:
            print(f"⚠ Could not generate PDF: {e}")

    return results


# ------------------------------------------------------------
# End of src/evaluation.py
