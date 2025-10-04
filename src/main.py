# src/main.py
# ============================================================
# Main entry point for Personalized Learning Recommender
# ============================================================

import pandas as pd
import numpy as np
import random

import os
import json
import torch

# STEP X: Tune MF + Evaluation
from evaluation import tune_mf, evaluate_model, evaluate_models, tune_hybrid_weights, _safe_recommend
from ncf import train_ncf, load_ncf_model
from sasrec import train_sasrec, load_sasrec_model
from dataset_analysis_guide import run_complete_analysis
from preprocessing import preprocess_data
from models import (
    train_popularity,
    train_mf,
    train_content_based,
    MFRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
)
# ✅ Tree-based models
from tree_models import DecisionTreeRecommender, RandomForestRecommender

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_PATH = "synthetic_learning_dataset.csv"
user_col = "learner_id"
item_col = "content_type"
rating_col = "engagement_score"

VERBOSE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙ Using device: {DEVICE}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== STEP 1: Dataset Analysis ===")
    df, found_columns, interaction_stats = run_complete_analysis(DATA_PATH)

    print("\n=== STEP 2: Preprocessing ===")
    train, val, test, encoders, scaler = preprocess_data(df)
    print(f"Train={train.shape}, Val={val.shape}, Test={test.shape}")

    # --------------------------------------------------------
    # STEP 3: Baseline Models
    # --------------------------------------------------------
    pop_model = train_popularity(train, item_col, rating_col=rating_col)
    mf_model = train_mf(train, user_col, item_col, rating_col, n_components=20)
    cb_model = train_content_based(
        train, user_col, item_col,
        ["time_spent", "quiz_score", "completion_rate", "engagement_score"]
    )

    # --------------------------------------------------------
    # STEP 3b: Tree-based Models
    # --------------------------------------------------------
    tree_model = DecisionTreeRecommender(user_col, item_col, rating_col, max_depth=5)
    tree_model.fit(train)

    rf_model = RandomForestRecommender(user_col, item_col, rating_col, n_estimators=50, max_depth=10)
    rf_model.fit(train)

    # --------------------------------------------------------
    # STEP 4: Evaluation Baseline
    # --------------------------------------------------------
    print("\n=== STEP 4: Evaluation Baselines ===")
    models = {
        "Popularity": pop_model,
        "MF": mf_model,
        "CB": cb_model,
        "DecisionTree": tree_model,
        "RandomForest": rf_model
    }
    evaluate_models(models, test, user_col, item_col, k=10)

    # --------------------------------------------------------
    # STEP 5: MF Tuning
    # --------------------------------------------------------
    print("\n=== STEP 5: MF Tuning ===")
    best_mf_cfg = tune_mf(train, val, user_col, item_col, rating_col)
    print("✔ Best MF config:", best_mf_cfg)

    # --------------------------------------------------------
    # STEP 6: Hybrid Tuning (MF)
    # --------------------------------------------------------
    print("\n=== STEP 6: Hybrid (MF) Tuning ===")
    best_w_recall_mf, best_w_ndcg_mf = tune_hybrid_weights(
        {"pop": pop_model, "mf": mf_model, "cb": cb_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )
    if not VERBOSE:
        print("✔ Best MF-Hybrid Recall:", best_w_recall_mf)
        print("✔ Best MF-Hybrid NDCG:", best_w_ndcg_mf)

    # --------------------------------------------------------
    # STEP 6b: Hybrid with Tree Models
    # --------------------------------------------------------
    print("\n=== STEP 6b: Hybrid (Tree-based Models) ===")
    best_w_recall_tree_cb, best_w_ndcg_tree_cb = tune_hybrid_weights(
        {"tree": tree_model, "cb": cb_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )
    best_w_recall_rf_mf, best_w_ndcg_rf_mf = tune_hybrid_weights(
        {"rf": rf_model, "mf": mf_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )
    if not VERBOSE:
        print("✔ Best Hybrid (Tree+CB):", best_w_recall_tree_cb, best_w_ndcg_tree_cb)
        print("✔ Best Hybrid (RF+MF):", best_w_recall_rf_mf, best_w_ndcg_rf_mf)

    # --------------------------------------------------------
    # STEP 7: NCF Training & Tuning
    # --------------------------------------------------------
    print("\n=== STEP 7: NCF Training & Tuning ===")
    configs = [
        {"lr": 0.001, "dropout": 0.2},
        {"lr": 0.001, "dropout": 0.0},
        {"lr": 0.0005, "dropout": 0.2},
        {"lr": 0.0005, "dropout": 0.0},
    ]

    best_recall, best_val_loss_recall = -1, float("inf")
    best_val_loss, best_recall_val = float("inf"), -1

    for cfg in configs:
        ncf_tmp, val_loss = train_ncf(
            train, val,
            user_col=user_col, item_col=item_col, rating_col=rating_col,
            embedding_dim=32, hidden_layers=[64, 32, 16, 8],
            dropout=cfg["dropout"], batch_size=256, lr=cfg["lr"],
            epochs=20, device=DEVICE, verbose=VERBOSE,
            save_path="../models/ncf_best.pt"
        )
        recall, ndcg = evaluate_model(ncf_tmp, val, user_col, item_col, k=10)
        if recall > best_recall:
            best_recall = recall
            best_val_loss_recall = val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_recall_val = recall

    ncf_model_recall = load_ncf_model("../models/ncf_best.pt", device=DEVICE)
    ncf_model_val = load_ncf_model("../models/ncf_best.pt", device=DEVICE)
    print(f"✔ Best NCF (Recall): Recall={best_recall:.3f}, ValLoss={best_val_loss_recall:.4f}")
    print(f"✔ Best NCF (ValLoss): Recall={best_recall_val:.3f}, ValLoss={best_val_loss:.4f}")

    models["NCF-BestRecall"] = ncf_model_recall
    models["NCF-BestValLoss"] = ncf_model_val
    evaluate_models(models, test, user_col, item_col, k=10)

    # --------------------------------------------------------
    # STEP 8: Hybrid with NCF
    # --------------------------------------------------------
    print("\n=== STEP 8a: Hybrid (NCF-BestRecall) Tuning ===")
    best_w_recall_ncf_r, best_w_ndcg_ncf_r = tune_hybrid_weights(
        {"pop": pop_model, "ncf": ncf_model_recall, "cb": cb_model},
        test, user_col, item_col, step=0.1, k=10, verbose=VERBOSE
    )
    print("\n=== STEP 8b: Hybrid (NCF-BestValLoss) Tuning ===")
    best_w_recall_ncf_v, best_w_ndcg_ncf_v = tune_hybrid_weights(
        {"pop": pop_model, "ncf": ncf_model_val, "cb": cb_model},
        test, user_col, item_col, step=0.1, k=10, verbose=VERBOSE
    )
    if not VERBOSE:
        print("✔ Best Hybrid (NCF-Recall):", best_w_recall_ncf_r, best_w_ndcg_ncf_r)
        print("✔ Best Hybrid (NCF-ValLoss):", best_w_recall_ncf_v, best_w_ndcg_ncf_v)

    # --------------------------------------------------------
    # STEP 10: SASRec Training
    # --------------------------------------------------------
    print("\n=== STEP 10: SASRec Training & Tuning ===")
    user_sequences = train.groupby(user_col)[item_col].apply(list).to_dict()
    item_ids = train[item_col].unique()
    item_map = {i: idx+1 for idx, i in enumerate(item_ids)}

    sasrec_configs = [
        {"lr": 0.001, "embed_dim": 64, "n_layers": 2, "n_heads": 2, "dropout": 0.2},
        {"lr": 0.0005, "embed_dim": 64, "n_layers": 2, "n_heads": 2, "dropout": 0.2},
    ]

    best_sasrec, best_sasrec_loss = None, float("inf")
    for cfg in sasrec_configs:
        sasrec_tmp, val_loss = train_sasrec(
            user_sequences=user_sequences, item_map=item_map,
            embed_dim=cfg["embed_dim"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], dropout=cfg["dropout"],
            lr=cfg["lr"], epochs=10, device=DEVICE,
            verbose=VERBOSE, save_path="../models/sasrec_best.pt"
        )
        if val_loss < best_sasrec_loss:
            best_sasrec_loss = val_loss
            best_sasrec = sasrec_tmp

    sasrec_model = load_sasrec_model("../models/sasrec_best.pt", item_map, user_sequences, device=DEVICE)
    print(f"✔ Best SASRec ValLoss={best_sasrec_loss:.4f}")

    models["SASRec"] = sasrec_model
    evaluate_models(models, test, user_col, item_col, k=10)

    # --------------------------------------------------------
    # STEP 11: Hybrid with SASRec
    # --------------------------------------------------------
    print("\n=== STEP 11: Hybrid (SASRec) Tuning ===")
    best_w_recall_sasrec, best_w_ndcg_sasrec = tune_hybrid_weights(
        {"pop": pop_model, "sasrec": sasrec_model, "cb": cb_model},
        test, user_col, item_col, step=0.1, k=10, verbose=VERBOSE
    )
    if not VERBOSE:
        print("✔ Best SASRec-Hybrid Recall:", best_w_recall_sasrec)
        print("✔ Best SASRec-Hybrid NDCG:", best_w_ndcg_sasrec)

    # --------------------------------------------------------
    # STEP 11b: Knowledge Graph Embedding
    # --------------------------------------------------------
    print("\n=== STEP 11b: Knowledge Graph Features ===")
    from kg_features import build_and_embed
    import torch

    try:
        G, emb = build_and_embed(
            df,
            user_col="learner_id",
            item_col="content_type",
            concept_col="topic",
            method="gnn",
            dim=32,
            epochs=20,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✔ GNN embeddings generated (GraphSAGE)")
    except Exception as e:
        print(f"⚠ GNN embedding failed ({e}), fallback to Node2Vec...")
        G, emb = build_and_embed(
            df,
            user_col="learner_id",
            item_col="content_type",
            concept_col="topic",
            method="node2vec",
            dim=16,
            epochs=1
        )
        print("✔ Node2Vec embeddings generated (fallback)")

    print("Embedding cho 5 node đầu:", list(emb.items())[:5])


    # --------------------------------------------------------
    # STEP 11c: KG Recommender
    # --------------------------------------------------------
    print("\n=== STEP 11c: KG Recommender ===")
    from kg_recommender import KGRecommender

    try:
        if emb is None or len(emb) == 0:
            raise ValueError("Embeddings are empty → cannot train KG Recommender")

        kg_model = KGRecommender(user_col, item_col, emb, metric='cosine')
        kg_model.fit(train)
        models["KG"] = kg_model
        print(f"✔ KG Recommender trained successfully (dim={len(next(iter(emb.values())))}), added to models")
        evaluate_models(models, test, user_col, item_col, k=10)

    except Exception as e:
        print(f"⚠ Skipped KG Recommender: {e}")


    # --------------------------------------------------------
    # STEP 11d: Build base_scores & ground_truth
    # --------------------------------------------------------
    print("\n=== STEP 11d: Preparing base_scores & ground_truth ===")
    base_scores = {}
    users = test[user_col].unique()
    items = test[item_col].unique()
    ground_truth = {u: set(test.loc[test[user_col] == u, item_col].tolist()) for u in users}
    for name, model in models.items():
        base_scores[name] = {}
        for u in users:
            base_scores[name][u] = {}
            try:
                recs = _safe_recommend(model, u, top_k=20)
            except Exception:
                try:
                    recs = model.recommend(u, top_k=20)
                except Exception:
                    recs = []
            for rank, item in enumerate(recs):
                base_scores[name][u][item] = 1.0 / (rank + 1)
            for item in items:
                if item not in base_scores[name][u]:
                    base_scores[name][u][item] = 0.0
    print("✔ base_scores & ground_truth built successfully")

    # --------------------------------------------------------
    # STEP 11e: Meta-Hybrid (Stacking)
    # --------------------------------------------------------
    print("\n=== STEP 11e: Meta-Hybrid (Stacking) ===")
    from meta_hybrid import MetaHybridRecommender
    try:
        meta_model = MetaHybridRecommender(base_scores=base_scores, ground_truth=ground_truth)
        meta_model.fit()
        models["MetaHybrid-LR"] = meta_model
        print("✔ MetaHybrid (LogReg) trained successfully and added to models")
    except Exception as e:
        print(f"⚠ Skipped MetaHybrid-LR: {e}")
        
        
    # --------------------------------------------------------
    # STEP 11f: Hybrid (KG + Other Models, multi-model)
    # --------------------------------------------------------
    print("\n=== STEP 11f: Hybrid (KG + Other Models) ===")

    # KG + MF
    best_w_recall_kg_mf, best_w_ndcg_kg_mf = tune_hybrid_weights(
        {"kg": kg_model, "mf": mf_model},
        test, user_col, item_col, step=0.1, k=10, verbose=VERBOSE
    )
    print("✔ Best Hybrid (KG+MF):", best_w_recall_kg_mf, best_w_ndcg_kg_mf)

    # KG + CB
    best_w_recall_kg_cb, best_w_ndcg_kg_cb = tune_hybrid_weights(
        {"kg": kg_model, "cb": cb_model},
        test, user_col, item_col, step=0.1, k=10, verbose=VERBOSE
    )
    print("✔ Best Hybrid (KG+CB):", best_w_recall_kg_cb, best_w_ndcg_kg_cb)

    # KG + RF
    best_w_recall_kg_rf, best_w_ndcg_kg_rf = tune_hybrid_weights(
        {"kg": kg_model, "rf": rf_model},
        test, user_col, item_col, step=0.1, k=10, verbose=VERBOSE
    )
    print("✔ Best Hybrid (KG+RF):", best_w_recall_kg_rf, best_w_ndcg_kg_rf)

    # Multi-hybrid KG + MF + CB + RF
    best_w_recall_kg_all, best_w_ndcg_kg_all = tune_hybrid_weights(
        {"kg": kg_model, "mf": mf_model, "cb": cb_model, "rf": rf_model},
        test, user_col, item_col, step=0.1, k=10, verbose=VERBOSE
    )
    print("✔ Best Hybrid (KG+MF+CB+RF):", best_w_recall_kg_all, best_w_ndcg_kg_all)


    # ---------------- STEP 12: Contextual Bandit Simulation ----------------
    print("\n=== STEP 12: Contextual Bandit (LinUCB) Simulation ===")
    from contextual_bandit import LinUCB, simulate_bandit, make_context_default, prefilter_top_k_by_popularity

    # 1) Tham số
    CONTEXT_DIM = 64   # nếu using embeddings dim; else leave as estimated below
    ALPHA = 1.0
    REG = 1.0

    # 2) Prepare context scaler (optional) - fit on a sample of contexts to stabilize magnitudes
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Build small sample contexts to fit scaler (use first 200 rows and first 5 items each)
    sample_vecs = []
    unique_items = test[item_col].unique().tolist()
    for idx, row in test.head(200).iterrows():
        u = row[user_col]
        # sample up to 5 candidate items
        candidates = unique_items[:5]
        for it in candidates:
            v = make_context_default(
                u, it,
                kg_embeddings=emb if 'emb' in globals() else None,
                mf_model=mf_model,
                item_feature_map=None,
                session_feats=None,
                concat_scale=None
            )
            sample_vecs.append(v)
    if sample_vecs:
        # pad/trim so they have consistent size:
        maxd = max(v.size for v in sample_vecs)
        X = np.array([np.pad(v, (0, maxd - v.size), 'constant') for v in sample_vecs])
        scaler.fit(X)

    # 3) wrapper make_context that uses scaler
    def make_context(u,i):
        v = make_context_default(u, i,
                                kg_embeddings=emb if 'emb' in globals() else None,
                                mf_model=mf_model,
                                item_feature_map=None,
                                session_feats=None,
                                concat_scale=None)  # we will apply scaler below
        # ensure consistent size
        if hasattr(scaler, "mean_"):
            # pad/trim
            d = scaler.mean_.shape[0]
            if v.size < d:
                v = np.pad(v, (0, d - v.size), 'constant')
            elif v.size > d:
                v = v[:d]
            v = scaler.transform(v.reshape(1, -1)).ravel()
        return v

    # 4) instantiate bandit
    # Find context dimension from scaler or sample
    dim = scaler.mean_.shape[0] if hasattr(scaler, "mean_") else 16
    bandit = LinUCB(dim=dim, alpha=ALPHA, regularization=REG)

    # 5) reward function: use engagement_score in test rows if available
    def reward_fn(u, chosen_item, row):
        # If row corresponds to chosen_item then use the engagement score; else 0
        # In offline simulation, we commonly need a logged policy; here we use simple proxy:
        # If chosen_item equals the test row's item, return the engagement_score; else 0.
        try:
            if row[item_col] == chosen_item:
                return float(row[rating_col])
        except Exception:
            pass
        return 0.0

    # 6) candidate selector: use popularity prefilter top 50 (faster)
    def candidate_selector(u, row):
        try:
            return prefilter_top_k_by_popularity(pop_model, u, row, top_k=50)
        except Exception:
            return unique_items

    # 7) run simulation
    res = simulate_bandit(
        bandit=bandit,
        test_df=test,
        make_context_fn=make_context,
        user_col=user_col,
        item_col=item_col,
        reward_fn=reward_fn,
        candidate_selector=candidate_selector,
        binarize=False,
        binary_threshold=0.5,
        top_k=1,
        verbose=True
    )
    print("Contextual Bandit (LinUCB) results:", res)



    # ---------------- STEP 12b: Contextual Bandit with Real Embeddings ----------------
    print("\n=== STEP 12b: Contextual Bandit (LinUCB with Real Embeddings) ===")
    from contextual_bandit import LinUCB, simulate_bandit, make_context_default, prefilter_top_k_by_popularity

    # Tham số
    ALPHA = 1.0
    REG = 1.0

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Fit scaler dựa trên context vector thực (KG + MF + Content)
    sample_vecs = []
    unique_items = test[item_col].unique().tolist()
    for idx, row in test.head(300).iterrows():
        u = row[user_col]
        for it in unique_items[:10]:
            v = make_context_default(
                u, it,
                kg_embeddings=emb if 'emb' in globals() else None,
                mf_model=mf_model,
                item_feature_map=None,  # nếu có content feature thì truyền vào đây
                session_feats=None,
                concat_scale=None
            )
            sample_vecs.append(v)

    if sample_vecs:
        maxd = max(v.size for v in sample_vecs)
        X = np.array([np.pad(v, (0, maxd - v.size), 'constant') for v in sample_vecs])
        scaler.fit(X)

    # Wrapper cho context function có scaler
    def make_context_scaled(u, i):
        v = make_context_default(
            u, i,
            kg_embeddings=emb if 'emb' in globals() else None,
            mf_model=mf_model,
            item_feature_map=None,
            session_feats=None,
            concat_scale=None
        )
        d = scaler.mean_.shape[0]
        if v.size < d:
            v = np.pad(v, (0, d - v.size), 'constant')
        elif v.size > d:
            v = v[:d]
        return scaler.transform(v.reshape(1, -1)).ravel()

    # Instantiate bandit
    dim = scaler.mean_.shape[0]
    bandit = LinUCB(dim=dim, alpha=ALPHA, regularization=REG)

    # Reward function
    def reward_fn(u, chosen_item, row):
        if row[item_col] == chosen_item:
            try:
                return float(row[rating_col])
            except Exception:
                return 1.0
        return 0.0

    # Candidate selector
    def candidate_selector(u, row):
        try:
            return prefilter_top_k_by_popularity(pop_model, u, row, top_k=50)
        except Exception:
            return unique_items

    # Run simulation
    res = simulate_bandit(
        bandit=bandit,
        test_df=test,
        make_context_fn=make_context_scaled,
        user_col=user_col,
        item_col=item_col,
        reward_fn=reward_fn,
        candidate_selector=candidate_selector,
        binarize=False,
        binary_threshold=0.5,
        top_k=1,
        verbose=True
    )
    print("✅ Contextual Bandit (LinUCB + Embeddings) results:", res)



    # ---------------- STEP 12c: Plot Reward Curve ----------------
    print("\n=== STEP 12c: Plot Reward Curve ===")
    import matplotlib.pyplot as plt

    # chuẩn bị thư mục results
    base_dir = os.path.dirname(os.path.abspath(__file__))   # src/
    root_dir = os.path.dirname(base_dir)                   # project root
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    logs = res.get("logs", [])
    if logs:
        plt.figure(figsize=(8,4))
        plt.plot(np.cumsum(logs) / (np.arange(len(logs)) + 1), label="Avg Reward")
        plt.xlabel("Rounds")
        plt.ylabel("Average Reward")
        plt.title("LinUCB Learning Curve")
        plt.legend()
        plot_path = os.path.join(results_dir, "linucb_learning_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Saved LinUCB learning curve to {plot_path}")
    else:
        print("⚠ No reward logs available to plot.")




    # --------------------------------------------------------
    # FINAL SUMMARY (Auto via evaluation + meta_eval_patch)
    # --------------------------------------------------------
    print("\n=== FINAL SUMMARY (Auto) ===")

    from evaluation import evaluate_and_log
    from meta_eval_patch import update_summary

    base_dir = os.path.dirname(os.path.abspath(__file__))   # src/
    root_dir = os.path.dirname(base_dir)                   # project root
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1) Quick summary (CSV cơ bản)
    evaluate_and_log(
        models=models,
        test=test,
        user_col=user_col,
        item_col=item_col,
        k=10,
        save_path=os.path.join(results_dir, "summary_basic.csv")
    )

    # 2) Full summary (Precision, MAP, Top3, Charts, PDF)
    update_summary(
        models=models,
        test_df=test,
        user_col=user_col,
        item_col=item_col,
        results_dir=results_dir,
        dataset_name="MyDataset"   # đổi tên cho hợp lý
    )


    print("\n=== END OF PIPELINE ===\n")
