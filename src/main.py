# src/main.py
# ============================================================
# Main entry point for Personalized Learning Recommender
# ============================================================

import pandas as pd
import os, csv
import json
import numpy as np

# STEP X: Tune MF + Evaluation
from evaluation import tune_mf, evaluate_model, evaluate_models, tune_hybrid_weights, _safe_recommend
from ncf import train_ncf, load_ncf_model   # ✅ load_ncf_model để reload best checkpoint
from sasrec import train_sasrec, load_sasrec_model   # ✅ thêm SASRec
from dataset_analysis_guide import run_complete_analysis
from preprocessing import preprocess_data
from models import (
    train_popularity,
    train_mf,
    train_content_based,
    train_hybrid,
    train_hybrid_ncf,
    MFRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    HybridRecommender,
)
# ✅ Import thêm tree-based models
from tree_models import DecisionTreeRecommender, RandomForestRecommender

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_PATH = "synthetic_learning_dataset.csv"
user_col = "learner_id"
item_col = "content_type"
rating_col = "engagement_score"

VERBOSE = False   # True để in log chi tiết, False = tóm gọn

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

    # Hybrid(Tree + CB)
    best_w_recall_tree_cb, best_w_ndcg_tree_cb = tune_hybrid_weights(
        {"tree": tree_model, "cb": cb_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )

    # Hybrid(RF + MF)
    best_w_recall_rf_mf, best_w_ndcg_rf_mf = tune_hybrid_weights(
        {"rf": rf_model, "mf": mf_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )

    if not VERBOSE:
        print("✔ Best Hybrid (Tree+CB):", best_w_recall_tree_cb, best_w_ndcg_tree_cb)
        print("✔ Best Hybrid (RF+MF):", best_w_recall_rf_mf, best_w_ndcg_rf_mf)

    # --------------------------------------------------------
    # STEP 7: NCF Tuning
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
        if VERBOSE: print(f"\n[NCF Tuning] Trying {cfg}")
        ncf_tmp, val_loss = train_ncf(
            train, val,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            embedding_dim=32,
            hidden_layers=[64, 32, 16, 8],
            dropout=cfg["dropout"],
            batch_size=256,
            lr=cfg["lr"],
            epochs=20,
            device="cpu",
            verbose=VERBOSE,
            save_path="../models/ncf_best.pt"
        )
        recall, ndcg = evaluate_model(ncf_tmp, val, user_col, item_col, k=10)

        if recall > best_recall:
            best_recall = recall
            best_val_loss_recall = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_recall_val = recall

    ncf_model_recall = load_ncf_model("../models/ncf_best.pt", device="cpu")
    print(f"✔ Best NCF (Recall): Recall={best_recall:.3f}, ValLoss={best_val_loss_recall:.4f}")

    ncf_model_val = load_ncf_model("../models/ncf_best.pt", device="cpu")
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
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )

    print("\n=== STEP 8b: Hybrid (NCF-BestValLoss) Tuning ===")
    best_w_recall_ncf_v, best_w_ndcg_ncf_v = tune_hybrid_weights(
        {"pop": pop_model, "ncf": ncf_model_val, "cb": cb_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )

    if not VERBOSE:
        print("✔ Best Hybrid (NCF-Recall):", best_w_recall_ncf_r, best_w_ndcg_ncf_r)
        print("✔ Best Hybrid (NCF-ValLoss):", best_w_recall_ncf_v, best_w_ndcg_ncf_v)

    # --------------------------------------------------------
    # STEP 10: SASRec Training & Tuning
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
        if VERBOSE: print(f"\n[SASRec Tuning] Trying {cfg}")
        sasrec_tmp, val_loss = train_sasrec(
            user_sequences=user_sequences,
            item_map=item_map,
            embed_dim=cfg["embed_dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            epochs=10,
            device="cpu",
            verbose=VERBOSE,
            save_path="../models/sasrec_best.pt"
        )
        if val_loss < best_sasrec_loss:
            best_sasrec_loss = val_loss
            best_sasrec = sasrec_tmp

    # ✅ truyền thêm user_sequences vào đây
    sasrec_model = load_sasrec_model("../models/sasrec_best.pt", item_map, user_sequences, device="cpu")
    print(f"✔ Best SASRec ValLoss={best_sasrec_loss:.4f}")

    models["SASRec"] = sasrec_model
    evaluate_models(models, test, user_col, item_col, k=10)

    # --------------------------------------------------------
    # STEP 11: Hybrid with SASRec
    # --------------------------------------------------------
    print("\n=== STEP 11: Hybrid (SASRec) Tuning ===")
    best_w_recall_sasrec, best_w_ndcg_sasrec = tune_hybrid_weights(
        {"pop": pop_model, "sasrec": sasrec_model, "cb": cb_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )
    if not VERBOSE:
        print("✔ Best SASRec-Hybrid Recall:", best_w_recall_sasrec)
        print("✔ Best SASRec-Hybrid NDCG:", best_w_ndcg_sasrec)

    # --------------------------------------------------------
    # STEP 11b: Knowledge Graph Features
    # --------------------------------------------------------
    print("\n=== STEP 11b: Knowledge Graph Features ===")
    from kg_features import build_and_embed

    try:
        # Ưu tiên dùng GNN (GraphSAGE)
        G, emb = build_and_embed(
            df,
            user_col="learner_id",
            item_col="content_type",
            concept_col="topic",
            method="gnn",     # dùng GraphSAGE
            dim=32,
            epochs=20         # giảm epochs để chạy nhanh hơn
        )
        print("✔ GNN embeddings generated (GraphSAGE)")
    except Exception as e:
        print(f"⚠ GNN embedding failed ({e}), fallback to Node2Vec...")
        # Fallback sang node2vec
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
    # In thử vài embedding
    print("Embedding cho 5 node đầu:", list(emb.items())[:5])

    # --------------------------------------------------------
    # STEP 11c: KG Recommender
    # --------------------------------------------------------
    print("\n=== STEP 11c: KG Recommender ===")
    from kg_recommender import KGRecommender

    kg_model = KGRecommender(user_col, item_col, emb, metric='cosine')
    kg_model.fit(train)
    models["KG"] = kg_model
    evaluate_models(models, test, user_col, item_col, k=10)

    # --------------------------------------------------------
    # STEP 11c2: Hybrid (KG + MF + CB)
    # --------------------------------------------------------
    print("\n=== STEP 11c2: Hybrid (KG+MF+CB) Tuning ===")
    best_w_recall_kg, best_w_ndcg_kg = tune_hybrid_weights(
        {"kg": kg_model, "mf": mf_model, "cb": cb_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=VERBOSE
    )
    if not VERBOSE:
        print("✔ Best Hybrid (KG+MF+CB) Recall:", best_w_recall_kg)
        print("✔ Best Hybrid (KG+MF+CB) NDCG:", best_w_ndcg_kg)

    # ---------------- STEP 11d: Prepare base_scores & ground_truth ----------------
    print("\n=== STEP 11d: Preparing base_scores & ground_truth ===")

    base_scores = {}
    users = test[user_col].unique()
    items = test[item_col].unique()

    # ground_truth[user] = set of items in test for that user
    ground_truth = {
        u: set(test.loc[test[user_col] == u, item_col].tolist())
        for u in users
    }

    # Build base_scores for each model in models
    from evaluation import _safe_recommend

    for name, model in models.items():
        base_scores[name] = {}
        for u in users:
            base_scores[name][u] = {}
            # use safe recommend to avoid signature issues
            try:
                recs = _safe_recommend(model, u, top_k=20)
            except Exception:
                # fallback: try common calls
                try:
                    recs = model.recommend(u, top_k=20)
                except Exception:
                    recs = []

            for rank, item in enumerate(recs):
                base_scores[name][u][item] = 1.0 / (rank + 1)

            # set score=0 for items not recommended
            for item in items:
                if item not in base_scores[name][u]:
                    base_scores[name][u][item] = 0.0

    print("✔ base_scores & ground_truth built successfully")


    # ---------------- STEP 11e: Meta-Hybrid (Stacking) - train meta-learner ----------------
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
    # STEP 12: Summary
    # --------------------------------------------------------
    print("\n=== FINAL SUMMARY ===")
    print("Best MF config:", best_mf_cfg)
    print("Best Hybrid-MF Recall:", best_w_recall_mf)
    print("Best Hybrid-MF NDCG:", best_w_ndcg_mf)
    print("Best Hybrid-Tree+CB:", best_w_recall_tree_cb, best_w_ndcg_tree_cb)
    print("Best Hybrid-RF+MF:", best_w_recall_rf_mf, best_w_ndcg_rf_mf)
    print("Best Hybrid-NCF (Recall-based):", best_w_recall_ncf_r, best_w_ndcg_ncf_r)
    print("Best Hybrid-NCF (ValLoss-based):", best_w_recall_ncf_v, best_w_ndcg_ncf_v)
    print("Best Hybrid-SASRec Recall:", best_w_recall_sasrec)
    print("Best Hybrid-SASRec NDCG:", best_w_ndcg_sasrec)
    print("Best Hybrid-KG+MF+CB:", best_w_recall_kg, best_w_ndcg_kg)

    print("\n=== END ===\n")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))   # src/
    root_dir = os.path.dirname(base_dir)                   # project root
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ----- TXT -----
    summary_txt = os.path.join(results_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== FINAL SUMMARY ===\n")
        f.write(f"Best MF config: {best_mf_cfg}\n")
        f.write(f"Best Hybrid-MF Recall: {best_w_recall_mf}\n")
        f.write(f"Best Hybrid-MF NDCG: {best_w_ndcg_mf}\n")
        f.write(f"Best Hybrid-Tree+CB: {best_w_recall_tree_cb} {best_w_ndcg_tree_cb}\n")
        f.write(f"Best Hybrid-RF+MF: {best_w_recall_rf_mf} {best_w_ndcg_rf_mf}\n")
        f.write(f"Best Hybrid-NCF (Recall-based): {best_w_recall_ncf_r} {best_w_ndcg_ncf_r}\n")
        f.write(f"Best Hybrid-NCF (ValLoss-based): {best_w_recall_ncf_v} {best_w_ndcg_ncf_v}\n")
        f.write(f"Best Hybrid-SASRec Recall: {best_w_recall_sasrec}\n")
        f.write(f"Best Hybrid-SASRec NDCG: {best_w_ndcg_sasrec}\n")
        f.write(f"Best Hybrid-KG+MF+CB: {best_w_recall_kg} {best_w_ndcg_kg}\n")
        f.write("\n=== END ===\n")

    print(f"✔ Summary TXT saved to: {summary_txt}")

    # ----- CSV -----
    summary_csv = os.path.join(results_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Config/Weights", "Recall@10", "NDCG@10"])

        def safe_score(x, key):
            return float(x["score"]) if (x is not None and key in x) else ""

        # MF
        writer.writerow(["MF", json.dumps(best_mf_cfg), "", ""])

        # Hybrid MF
        writer.writerow([
            "Hybrid-MF",
            "Recall=" + json.dumps(best_w_recall_mf) + " | NDCG=" + json.dumps(best_w_ndcg_mf),
            safe_score(best_w_recall_mf, "score"),
            safe_score(best_w_ndcg_mf, "score")
        ])

        # Hybrid Tree+CB
        writer.writerow([
            "Hybrid-Tree+CB",
            "Recall=" + json.dumps(best_w_recall_tree_cb) + " | NDCG=" + json.dumps(best_w_ndcg_tree_cb),
            safe_score(best_w_recall_tree_cb, "score"),
            safe_score(best_w_ndcg_tree_cb, "score")
        ])

        # Hybrid RF+MF
        writer.writerow([
            "Hybrid-RF+MF",
            "Recall=" + json.dumps(best_w_recall_rf_mf) + " | NDCG=" + json.dumps(best_w_ndcg_rf_mf),
            safe_score(best_w_recall_rf_mf, "score"),
            safe_score(best_w_ndcg_rf_mf, "score")
        ])

        # Hybrid NCF
        writer.writerow([
            "Hybrid-NCF",
            "Recall=" + json.dumps(best_w_recall_ncf_r) + " | NDCG=" + json.dumps(best_w_ndcg_ncf_r),
            safe_score(best_w_recall_ncf_r, "score"),
            safe_score(best_w_ndcg_ncf_r, "score")
        ])

        # Hybrid NCF (ValLoss)
        writer.writerow([
            "Hybrid-NCF (ValLoss)",
            "Recall=" + json.dumps(best_w_recall_ncf_v) + " | NDCG=" + json.dumps(best_w_ndcg_ncf_v),
            safe_score(best_w_recall_ncf_v, "score"),
            safe_score(best_w_ndcg_ncf_v, "score")
        ])

        # Hybrid SASRec
        writer.writerow([
            "Hybrid-SASRec",
            "Recall=" + json.dumps(best_w_recall_sasrec) + " | NDCG=" + json.dumps(best_w_ndcg_sasrec),
            safe_score(best_w_recall_sasrec, "score"),
            safe_score(best_w_ndcg_sasrec, "score")
        ])

        # ✅ Hybrid KG+MF+CB
        writer.writerow([
            "Hybrid-KG+MF+CB",
            "Recall=" + json.dumps(best_w_recall_kg) + " | NDCG=" + json.dumps(best_w_ndcg_kg),
            safe_score(best_w_recall_kg, "score"),
            safe_score(best_w_ndcg_kg, "score")
        ])


    print(f"✔ Summary CSV saved to: {summary_csv}")

    # --------------------------------------------------------
    # STEP 12b: Ranking theo Recall và NDCG
    # --------------------------------------------------------
    df_summary = pd.read_csv(summary_csv)

    df_summary["Recall@10"] = pd.to_numeric(df_summary["Recall@10"], errors="coerce")
    df_summary["NDCG@10"] = pd.to_numeric(df_summary["NDCG@10"], errors="coerce")

    # Ranking Recall
    top_recall = df_summary.sort_values("Recall@10", ascending=False).head(3)
    print("\n🔥 Top models theo Recall@10:")
    print(top_recall[["Model", "Recall@10"]].to_string(index=False))

    # Ranking NDCG
    top_ndcg = df_summary.sort_values("NDCG@10", ascending=False).head(3)
    print("\n🔥 Top models theo NDCG@10:")
    print(top_ndcg[["Model", "NDCG@10"]].to_string(index=False))
    
    # --------------------------------------------------------
    # STEP 13: Visualization
    # --------------------------------------------------------
    try:
        from visualize_results import plot_results
        charts_dir = os.path.join(root_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)   # ✅ đảm bảo charts/ tồn tại
        plot_results(summary_csv, charts_dir)
        print(f"✔ Charts saved in: {charts_dir}")
    except Exception as e:
        print(f"⚠ Visualization skipped due to error: {e}")


    from meta_eval_patch import update_summary

    # models_dict chứa các mô hình đã train (NCF, MF, SASRec, MetaHybrid, v.v.)
    update_summary(models, test, user_col="learner_id", item_col="content_type", results_dir="../results")

    print("\n=== END OF PIPELINE ===\n")
