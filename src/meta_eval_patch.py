# meta_eval_patch.py
# ============================================================
# Cập nhật summary.csv sau khi pipeline chạy xong
# - Tự động tính Recall@10, NDCG@10 của tất cả model trong dict
# - Gộp kết quả vào file summary.csv
# - Sắp xếp và thêm cột Rank
# - Vẽ thêm biểu đồ Grouped Top 3 (Recall vs NDCG)
# - Xuất bảng Top 3 ra top3.csv
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import evaluate_model


def update_summary(models, test_df, user_col="learner_id", item_col="content_type", K=10, results_dir="../results"):
    os.makedirs(results_dir, exist_ok=True)
    charts_dir = os.path.join(results_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.csv")
    top3_path = os.path.join(results_dir, "top3.csv")

    results = []
    for name, model in models.items():
        try:
            recall, ndcg = evaluate_model(model, test_df, user_col, item_col, k=K)
            model_type = infer_model_type(name)
            results.append({
                "Model": name,
                "Type": model_type,
                "Recall@10": round(recall, 6),
                "NDCG@10": round(ndcg, 6)
            })
            print(f"✔ Evaluated {name}: Recall@10={recall:.4f}, NDCG@10={ndcg:.4f}")
        except Exception as e:
            print(f"⚠ Skipped model {name} due to error: {e}")

    df = pd.DataFrame(results)
    df["RankRecall"] = df["Recall@10"].rank(ascending=False, method="min").astype(int)
    df["RankNDCG"] = df["NDCG@10"].rank(ascending=False, method="min").astype(int)
    df.sort_values(by="Recall@10", ascending=False, inplace=True)

    df.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV updated: {summary_path}")

    # In ra Top 3
    top3_recall = df.head(3)
    top3_ndcg = df.sort_values("NDCG@10", ascending=False).head(3)

    # Union để lấy đầy đủ rồi chọn top 3 theo Recall
    top_models = pd.concat([top3_recall, top3_ndcg]).drop_duplicates(subset=["Model"])
    top_models = top_models.sort_values("Recall@10", ascending=False).head(3)

    print("\n🔥 Top 3 theo Recall@10:")
    print(top3_recall[["Model", "Recall@10"]].to_string(index=False))
    print("\n🔥 Top 3 theo NDCG@10:")
    print(top3_ndcg[["Model", "NDCG@10"]].to_string(index=False))

    # --- Xuất bảng Top 3 ra CSV ---
    df_top3_export = top_models[["Model", "Recall@10", "NDCG@10"]]
    df_top3_export.to_csv(top3_path, index=False)
    print(f"✅ Top 3 CSV saved: {top3_path}")

    # --- Vẽ biểu đồ Grouped ---
    plot_top3_grouped(top_models, charts_dir, "top3_grouped.png")

    return df


def plot_top3_grouped(df_top, charts_dir, filename):
    """Vẽ biểu đồ grouped (Recall vs NDCG) cho Top 3"""
    x = range(len(df_top))
    width = 0.35

    plt.figure(figsize=(7, 5))
    bars1 = plt.bar([p - width/2 for p in x], df_top["Recall@10"], width, label="Recall@10", color="steelblue", alpha=0.85)
    bars2 = plt.bar([p + width/2 for p in x], df_top["NDCG@10"], width, label="NDCG@10", color="orange", alpha=0.85)

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.annotate(f"{height:.3f}",
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    plt.xticks(x, df_top["Model"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Top 3 Models by Recall & NDCG")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(charts_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved Grouped Top 3 chart: {save_path}")


def infer_model_type(name: str) -> str:
    name_lower = name.lower()
    if "meta" in name_lower:
        return "MetaHybrid"
    if "hybrid" in name_lower:
        return "Hybrid"
    if "ncf" in name_lower:
        return "Single"
    if "sasrec" in name_lower:
        return "Single"
    if "randomforest" in name_lower or "decisiontree" in name_lower:
        return "Single"
    if "kg" in name_lower:
        return "KG"
    if "mf" in name_lower:
        return "Single"
    if "cb" in name_lower:
        return "Single"
    if "pop" in name_lower:
        return "Single"
    return "Single"


if __name__ == "__main__":
    print("⚠ This module is meant to be imported and used in main.py, not run directly.")
