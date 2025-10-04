# meta_eval_patch.py
# ============================================================
# Cập nhật summary sau khi pipeline chạy xong
# - Tính Recall@10, NDCG@10, Precision@10, MAP@10
# - Xuất summary.csv, summary.txt, top3.csv
# - Thêm cột Rank
# - Vẽ biểu đồ Top 3
# - Xuất PDF report (có tên dataset)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import evaluate_model, _safe_recommend
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ================= Metric Helpers =================
def precision_at_k(recommended, relevant, k=10):
    if not recommended: return 0.0
    recommended_at_k = recommended[:k]
    hits = sum(1 for i in recommended_at_k if i in relevant)
    return hits / k

def average_precision(recommended, relevant, k=10):
    if not recommended: return 0.0
    score, hits = 0.0, 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0

def evaluate_model_extended(model, test_df, user_col, item_col, k=10):
    """Tính Recall, NDCG, Precision, MAP cho 1 model"""
    recall, ndcg = evaluate_model(model, test_df, user_col, item_col, k=k)
    users = test_df[user_col].unique()
    precs, maps = [], []
    for u in users:
        try:
            recs = _safe_recommend(model, u, top_k=k)
        except Exception:
            recs = []
        relevant = set(test_df.loc[test_df[user_col] == u, item_col])
        precs.append(precision_at_k(recs, relevant, k))
        maps.append(average_precision(recs, relevant, k))
    precision = sum(precs) / len(precs) if precs else 0.0
    map_score = sum(maps) / len(maps) if maps else 0.0
    return recall, ndcg, precision, map_score

# ================= Update Summary =================
def update_summary(models, test_df, user_col="learner_id", item_col="content_type",
                   K=10, results_dir="../results", dataset_name="UnknownDataset"):
    os.makedirs(results_dir, exist_ok=True)
    charts_dir = os.path.join(results_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.csv")
    top3_path = os.path.join(results_dir, "top3.csv")
    summary_txt = os.path.join(results_dir, "summary.txt")
    summary_pdf = os.path.join(results_dir, "summary_report.pdf")

    results = []
    for name, model in models.items():
        try:
            recall, ndcg, prec, map_score = evaluate_model_extended(model, test_df, user_col, item_col, k=K)
            results.append({
                "Model": name,
                "Type": infer_model_type(name),
                "Recall@10": round(recall, 6),
                "NDCG@10": round(ndcg, 6),
                "Precision@10": round(prec, 6),
                "MAP@10": round(map_score, 6)
            })
            print(f"✔ Evaluated {name}: R={recall:.4f}, N={ndcg:.4f}, P={prec:.4f}, MAP={map_score:.4f}")
        except Exception as e:
            print(f"⚠ Skipped model {name} due to error: {e}")

    df = pd.DataFrame(results)
    df["RankRecall"] = df["Recall@10"].rank(ascending=False, method="min").astype(int)
    df["RankNDCG"] = df["NDCG@10"].rank(ascending=False, method="min").astype(int)
    df.sort_values(by="Recall@10", ascending=False, inplace=True)

    # Xuất summary.csv
    df.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV updated: {summary_path}")

    # Top 3
    top3_recall = df.head(3)
    top3_ndcg = df.sort_values("NDCG@10", ascending=False).head(3)
    top_models = pd.concat([top3_recall, top3_ndcg]).drop_duplicates(subset=["Model"])
    top_models = top_models.sort_values("Recall@10", ascending=False).head(3)

    print("\n🔥 Top 3 theo Recall@10:")
    print(top3_recall[["Model", "Recall@10"]].to_string(index=False))
    print("\n🔥 Top 3 theo NDCG@10:")
    print(top3_ndcg[["Model", "NDCG@10"]].to_string(index=False))

    # Xuất top3.csv
    top_models[["Model", "Recall@10", "NDCG@10", "Precision@10", "MAP@10"]].to_csv(top3_path, index=False)
    print(f"✅ Top 3 CSV saved: {top3_path}")

    # Xuất summary.txt
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"=== FINAL SUMMARY ({dataset_name}) ===\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n=== TOP 3 MODELS ===\n")
        f.write(top_models.to_string(index=False))
        f.write("\n")
    print(f"✅ Summary TXT saved: {summary_txt}")

    # Vẽ charts
    chart2 = os.path.join(charts_dir, "top3_grouped2.png")
    chart4 = os.path.join(charts_dir, "top3_grouped4.png")
    plot_top3_grouped2(top_models, charts_dir, "top3_grouped2.png")
    plot_top3_grouped4(top_models, charts_dir, "top3_grouped4.png")

    # Xuất PDF report
    generate_pdf_report(df, top_models, summary_pdf, chart2, chart4, dataset_name)

    return df, top_models

# ================= PDF Report =================
def generate_pdf_report(df, top_models, filename, chart2, chart4, dataset_name):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph(f"📊 Recommendation Models Evaluation Report ({dataset_name})", styles["Title"]))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph("=== Full Summary ===", styles["Heading2"]))
    table_data = [list(df.columns)] + df.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("=== Top 3 Models ===", styles["Heading2"]))
    top_data = [list(top_models.columns)] + top_models.values.tolist()
    top_table = Table(top_data, repeatRows=1)
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
    ]))
    flow.append(top_table)
    flow.append(Spacer(1, 20))

    flow.append(Paragraph("=== Charts ===", styles["Heading2"]))
    if os.path.exists(chart2):
        flow.append(Image(chart2, width=400, height=250))
    if os.path.exists(chart4):
        flow.append(Image(chart4, width=400, height=300))

    doc.build(flow)
    print(f"📄 PDF report generated: {filename}")

# ================= Plotting =================
def plot_top3_grouped2(df_top, charts_dir, filename):
    x = range(len(df_top))
    width = 0.35
    plt.figure(figsize=(7, 5))
    bars1 = plt.bar([p - width/2 for p in x], df_top["Recall@10"], width, label="Recall@10", alpha=0.85)
    bars2 = plt.bar([p + width/2 for p in x], df_top["NDCG@10"], width, label="NDCG@10", alpha=0.85)
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, df_top["Model"], rotation=30, ha="right")
    plt.ylabel("Score"); plt.title("Top 3 Models by Recall & NDCG"); plt.legend(); plt.tight_layout()
    save_path = os.path.join(charts_dir, filename)
    plt.savefig(save_path); plt.close()
    print(f"📊 Saved Top 3 chart (Recall+NDCG): {save_path}")

def plot_top3_grouped4(df_top, charts_dir, filename):
    x = range(len(df_top)); width = 0.2
    plt.figure(figsize=(9, 6))
    bars1 = plt.bar([p - 1.5*width for p in x], df_top["Recall@10"], width, label="Recall@10", alpha=0.85)
    bars2 = plt.bar([p - 0.5*width for p in x], df_top["NDCG@10"], width, label="NDCG@10", alpha=0.85)
    bars3 = plt.bar([p + 0.5*width for p in x], df_top["Precision@10"], width, label="Precision@10", alpha=0.85)
    bars4 = plt.bar([p + 1.5*width for p in x], df_top["MAP@10"], width, label="MAP@10", alpha=0.85)
    for bar in bars1 + bars2 + bars3 + bars4:
        height = bar.get_height()
        plt.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, df_top["Model"], rotation=30, ha="right")
    plt.ylabel("Score"); plt.title("Top 3 Models by Recall, NDCG, Precision & MAP"); plt.legend(); plt.tight_layout()
    save_path = os.path.join(charts_dir, filename)
    plt.savefig(save_path); plt.close()
    print(f"📊 Saved Top 3 chart (4 metrics): {save_path}")

# ================= Utils =================
def infer_model_type(name: str) -> str:
    name_lower = name.lower()
    if "meta" in name_lower: return "MetaHybrid"
    if "hybrid" in name_lower: return "Hybrid"
    if "kg" in name_lower: return "KG"
    if any(x in name_lower for x in ["ncf", "sasrec", "mf", "cb", "pop", "randomforest", "decisiontree"]):
        return "Single"
    return "Single"

if __name__ == "__main__":
    print("⚠ This module is meant to be imported and used in main.py, not run directly.")
