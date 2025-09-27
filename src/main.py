# src/main.py
# ============================================================
# Main entry point for Personalized Learning Recommender
# ============================================================

import pandas as pd
from dataset_analysis_guide import run_complete_analysis

from evaluation import tune_hybrid_weights

from preprocessing import preprocess_data
from models import (
    train_popularity,
    train_mf,
    train_content_based,
    train_hybrid,
)
from evaluation import evaluate_models


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_PATH = "synthetic_learning_dataset.csv"
user_col = "learner_id"
item_col = "content_type"
rating_col = "engagement_score"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== STEP 1: Dataset Analysis ===")
    # lấy dataset từ file
    df, found_columns, interaction_stats = run_complete_analysis(DATA_PATH)

    print("\n=== STEP 2: Preprocessing ===")
    # chỉ truyền df, không truyền user_col/item_col/rating_col
    train, val, test, encoders, scaler = preprocess_data(df)

    print(f"Train shape: {train.shape}")
    print(f"Val shape:   {val.shape}")
    print(f"Test shape:  {test.shape}")
    print("Train head:")
    print(train.head(), "\n")

    # --------------------------------------------------------
    # STEP 3: Baseline Models
    # --------------------------------------------------------
    print("=== STEP 3: Baseline Models ===")
    pop_model = train_popularity(train, item_col, rating_col=rating_col)
    mf_model = train_mf(train, user_col, item_col, rating_col, n_components=20)

    # --------------------------------------------------------
    # STEP 4: Demo Recommendations
    # --------------------------------------------------------
    print("\n=== STEP 4: Demo Recommendation ===")
    sample_user = train[user_col].iloc[0]
    print("Top-5 Popular items:", pop_model.recommend(user_id=sample_user, top_k=10))
    print("Top-5 MF for user:", mf_model.recommend(user_id=sample_user, top_k=10))

    # --------------------------------------------------------
    # STEP 5: Content-Based & Hybrid
    # --------------------------------------------------------
    print("\n=== STEP 5: Content-Based & Hybrid Models ===")
    feature_cols = ["time_spent", "quiz_score", "completion_rate", "engagement_score"]

    cb_model = train_content_based(train, user_col, item_col, feature_cols)
    print("Top-5 CB for sample user:", cb_model.recommend(user_id=sample_user, top_k=10))

    hybrid_model = train_hybrid(
        {"pop": pop_model, "mf": mf_model, "cb": cb_model},
        weights={"pop": 0.2, "mf": 0.2, "cb": 0.6},  # ưu tiên Content-Based
    )

    print("Top-5 Hybrid for sample user:", hybrid_model.recommend(user_id=sample_user, top_k=10))

    # --------------------------------------------------------
    # STEP 6: Evaluation
    # --------------------------------------------------------
    print("\n=== STEP 6: Evaluation Framework ===")
    models = {
        "Popularity": pop_model,
        "MF": mf_model,
        "CB": cb_model,
        "Hybrid": hybrid_model,
    }
    evaluate_models(models, test, user_col, item_col, k=10)

    print("\n=== STEP 7: Tune Hybrid Weights ===")
    best_w_recall, best_w_ndcg = tune_hybrid_weights(
        {"pop": pop_model, "mf": mf_model, "cb": cb_model},
        test, user_col, item_col,
        step=0.1, k=10, verbose=False
    )

    print("Best weights (Recall@5):", best_w_recall)
    print("Best weights (NDCG@5):", best_w_ndcg)


    
    print("\n=== END OF PIPELINE ===\n")
