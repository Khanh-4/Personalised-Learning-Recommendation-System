# main.py
# ============================================================
# Orchestrates the whole pipeline:
# 1. Dataset analysis (EDA)
# 2. Preprocessing
# 3. Baseline models (Popularity, MF)
# 4. Evaluation
# ============================================================

from dataset_analysis_guide import run_complete_analysis
from preprocessing import preprocess_data
from models import train_popularity, train_mf, train_content_based, train_hybrid
# sau này sẽ import thêm models & evaluation

# ------------------------------------------------------------
# 1. DATASET ANALYSIS (EDA)
# ------------------------------------------------------------
file_path = "../dataset/personalized_learning_dataset.csv"

print("\n=== STEP 1: Dataset Analysis ===")
df, found_columns, interaction_stats = run_complete_analysis(file_path)
print("\n=== STEP 1 DONE: Dataset Analysis Finished ===")


# ------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------
print("\n=== STEP 2: Preprocessing ===")
user_col = "learner_id"
item_col = "content_type"
rating_col = "engagement_score"

train, val, test, encoders, scaler = preprocess_data(
    df,
    user_col=user_col,
    item_col=item_col,
    rating_col=rating_col
)

print("Train shape:", train.shape)
print("Val shape:", val.shape)
print("Test shape:", test.shape)
print("Train head:\n", train.head())

# ------------------------------------------------------------
# 3. BASELINE MODELS
# ------------------------------------------------------------
print("\n=== STEP 3: Baseline Models ===")
pop_model = train_popularity(train, item_col, rating_col)
mf_model = train_mf(train, user_col, item_col, rating_col, n_components=20)

# ------------------------------------------------------------
# 4. DEMO RECOMMENDATION
# ------------------------------------------------------------
print("\n=== STEP 4: Demo Recommendation ===")
print("Top-5 Popular items:", pop_model.recommend(top_k=5))
print("Top-5 MF for user 1:", mf_model.recommend(user_id=1, top_k=5))

# ------------------------------------------------------------
# 5. Content-Based + Hybrid
# ------------------------------------------------------------
print("\n=== STEP 5: Content-Based & Hybrid Models ===")

feature_cols = ["time_spent", "quiz_score", "completion_rate", "engagement_score"]
cb_model = train_content_based(train, user_col, item_col, feature_cols)

hybrid_model = train_hybrid(
    {"pop": pop_model, "mf": mf_model, "cb": cb_model},
    weights={"pop": 0.2, "mf": 0.5, "cb": 0.3}
)

# chọn user thật từ dataset
# chọn user có trong train (đảm bảo CB model biết user này)
sample_user = train[user_col].iloc[0]

print(f"Top-5 CB for {sample_user}:", cb_model.recommend(user_id=sample_user, top_k=5))
print(f"Top-5 Hybrid for {sample_user}:", hybrid_model.recommend(user_id=sample_user, top_k=5))

print("\n=== ALL STEPS COMPLETED ===")
# ------------------------------------------------------------
# END OF FILE