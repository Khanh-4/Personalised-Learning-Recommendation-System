# visualize_results.py
# ============================================================
# Sinh biểu đồ đánh giá kết quả (Recall / NDCG) từ summary.csv
# + Xuất PDF Report có Ranking
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def plot_results(summary_csv, charts_dir, pdf_report=True):
    # Load summary.csv
    df_summary = pd.read_csv(summary_csv)

    # Đảm bảo thư mục charts tồn tại
    os.makedirs(charts_dir, exist_ok=True)

    charts = []  # lưu lại path hình ảnh để nhúng vào PDF

    # --------------------------------------------------------
    # Biểu đồ Recall riêng
    # --------------------------------------------------------
    plt.figure(figsize=(10,6))
    bars = plt.bar(df_summary["Model"], df_summary["Recall@10"], color="steelblue", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, color="blue")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Recall@10")
    plt.title("Recall Comparison Across Models")
    plt.tight_layout()
    recall_chart_path = os.path.join(charts_dir, "recall_comparison.png")
    plt.savefig(recall_chart_path)
    plt.close()
    charts.append(recall_chart_path)

    # --------------------------------------------------------
    # Biểu đồ NDCG riêng
    # --------------------------------------------------------
    plt.figure(figsize=(10,6))
    bars = plt.bar(df_summary["Model"], df_summary["NDCG@10"], color="orange", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, color="darkorange")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("NDCG@10")
    plt.title("NDCG Comparison Across Models")
    plt.tight_layout()
    ndcg_chart_path = os.path.join(charts_dir, "ndcg_comparison.png")
    plt.savefig(ndcg_chart_path)
    plt.close()
    charts.append(ndcg_chart_path)

    # --------------------------------------------------------
    # Biểu đồ Grouped (Recall + NDCG)
    # --------------------------------------------------------
    x = range(len(df_summary))
    width = 0.35
    plt.figure(figsize=(12,6))
    bars1 = plt.bar([p - width/2 for p in x], df_summary["Recall@10"], width, label="Recall@10", color="steelblue", alpha=0.85)
    bars2 = plt.bar([p + width/2 for p in x], df_summary["NDCG@10"], width, label="NDCG@10", color="orange", alpha=0.85)
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, color="blue")
    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, color="darkorange")
    plt.xticks(x, df_summary["Model"], rotation=30, ha="right")
    plt.ylabel("Scores")
    plt.title("Recall vs NDCG Comparison Across Models")
    plt.legend()
    plt.tight_layout()
    grouped_chart_path = os.path.join(charts_dir, "recall_ndcg_comparison.png")
    plt.savefig(grouped_chart_path)
    plt.close()
    charts.append(grouped_chart_path)

    print(f"✔ Charts saved in {charts_dir}")

    # --------------------------------------------------------
    # Xuất PDF Report
    # --------------------------------------------------------
    if pdf_report:
        pdf_path = os.path.join(charts_dir, "evaluation_report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        flow = []

        flow.append(Paragraph("Evaluation Report", styles["Title"]))
        flow.append(Spacer(1, 12))
        flow.append(Paragraph("This report summarizes model evaluation results including Recall@10 and NDCG@10.", styles["Normal"]))
        flow.append(Spacer(1, 12))

        # Bảng kết quả
        data = [df_summary.columns.tolist()] + df_summary.values.tolist()
        table = Table(data)
        flow.append(Paragraph("📊 Full Results", styles["Heading2"]))
        flow.append(table)
        flow.append(Spacer(1, 20))

        # Ranking Top 3
        top_recall = df_summary.sort_values("Recall@10", ascending=False).head(3)[["Model", "Recall@10"]]
        top_ndcg = df_summary.sort_values("NDCG@10", ascending=False).head(3)[["Model", "NDCG@10"]]

        flow.append(Paragraph("🔥 Top 3 Models by Recall@10", styles["Heading2"]))
        data_recall = [top_recall.columns.tolist()] + top_recall.values.tolist()
        flow.append(Table(data_recall))
        flow.append(Spacer(1, 12))

        flow.append(Paragraph("🔥 Top 3 Models by NDCG@10", styles["Heading2"]))
        data_ndcg = [top_ndcg.columns.tolist()] + top_ndcg.values.tolist()
        flow.append(Table(data_ndcg))
        flow.append(Spacer(1, 20))


        # Chèn các biểu đồ
        for chart in charts:
            flow.append(Paragraph(os.path.basename(chart), styles["Heading2"]))
            flow.append(Image(chart, width=400, height=250))
            flow.append(Spacer(1, 20))

        doc.build(flow)
        print(f"✔ PDF report generated: {pdf_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))   # src/
    root_dir = os.path.dirname(base_dir)                   # project root
    results_dir = os.path.join(root_dir, "results")
    charts_dir = os.path.join(root_dir, "charts")

    summary_csv = os.path.join(results_dir, "summary.csv")
    plot_results(summary_csv, charts_dir, pdf_report=True)
