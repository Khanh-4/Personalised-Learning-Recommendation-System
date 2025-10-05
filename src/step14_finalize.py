# src/step14_finalize.py
"""
Step 14: Finalize pipeline
 - Export final CSV / TXT / PDF summary (uses existing evaluate_and_log / meta_eval_patch)
 - Dashboard / overview charts
 - Bandit parameter sweep (LinUCB alpha grid, EpsilonGreedy epsilons)
 - Simple ablation study for contextual features (full vs -KG vs -MF)
Assumes the main pipeline already created:
 - models (dict), pop_model, mf_model, emb (kg embeddings dict or None)
 - test (DataFrame), user_col, item_col, rating_col
 - results_dir (string)
 - functions: simulate_bandit, make_context_default, LinUCB, EpsilonGreedy, prefilter_top_k_by_popularity
 - evaluate_and_log, update_summary available
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# IMPORTS expected to exist in project
from contextual_bandit import simulate_bandit, make_context_default, LinUCB, EpsilonGreedy, prefilter_top_k_by_popularity
from evaluation import evaluate_and_log
from meta_eval_patch import update_summary

# -------------------------
# Helper: safe plot save
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Part A: Final exports (CSV/TXT/PDF)
# -------------------------
def export_final_reports(models, test_df, user_col, item_col, rating_col, results_dir, dataset_name="Dataset"):
    ensure_dir(results_dir)
    # 1) quick metrics CSV via evaluation.evaluate_and_log (already prints & saves)
    basic_csv = os.path.join(results_dir, "summary_basic.csv")
    try:
        evaluate_and_log(models=models, test=test_df, user_col=user_col, item_col=item_col, k=10, save_path=basic_csv)
    except Exception as e:
        print("⚠ evaluate_and_log failed:", e)

    # 2) full summary via meta_eval_patch
    try:
        update_summary(models=models, test_df=test_df, user_col=user_col, item_col=item_col, results_dir=results_dir, dataset_name=dataset_name)
    except Exception as e:
        print("⚠ update_summary failed:", e)

# -------------------------
# Part B: Dashboard / overview
# -------------------------
def build_dashboard(results_dir):
    charts_dir = os.path.join(results_dir, "charts")
    ensure_dir(charts_dir)
    # expected files from earlier steps: top3_grouped2.png, top3_grouped4.png, linucb_learning_curve.png, charts/bandit_comparison.png
    imgs = []
    for fn in ["charts/top3_grouped2.png", "charts/top3_grouped4.png", "linucb_learning_curve.png", "charts/bandit_comparison.png"]:
        p = os.path.join(results_dir, fn)
        if os.path.exists(p):
            imgs.append(p)

    # Make a simple PDF overview containing these images
    pdf_path = os.path.join(results_dir, "final_overview.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph("Final Dashboard Overview", styles["Title"]))
    flow.append(Spacer(1, 8))
    for p in imgs:
        flow.append(Paragraph(os.path.basename(p), styles["Heading3"]))
        try:
            img = Image(p, width=400, height=250)
            flow.append(img)
            flow.append(Spacer(1, 10))
        except Exception as e:
            print("⚠ Could not add image to PDF:", p, e)
    doc.build(flow)
    print("📊 Dashboard PDF:", pdf_path)
    return pdf_path

# -------------------------
# Part C: Bandit parameter sweep
# -------------------------
def bandit_parameter_sweep(test_df, user_col, item_col, rating_col, results_dir,
                           pop_model=None, mf_model=None, emb=None,
                           make_context_fn=None, candidate_selector=None,
                           n_samples=500):
    """
    Sweep grid:
     - LinUCB alpha grid (e.g. [0.1, 0.5, 1.0, 2.0])
     - EpsilonGreedy epsilons (e.g. [0.01, 0.05, 0.1, 0.2])
    Save CSV + plot.
    """
    charts_dir = os.path.join(results_dir, "charts")
    ensure_dir(charts_dir)

    lin_alphas = [0.1, 0.5, 1.0, 2.0]
    epsilons = [0.01, 0.05, 0.1, 0.2]

    records = []
    unique_items = test_df[item_col].unique().tolist()

    # default candidate selector
    if candidate_selector is None:
        def candidate_selector(u, row): 
            try:
                return prefilter_top_k_by_popularity(pop_model, u, row, top_k=50)
            except Exception:
                return unique_items

    # LinUCB sweep
    for a in lin_alphas:
        dim = make_context_fn( next(iter(test_df[user_col].unique())), next(iter(unique_items)) ).shape[0]
        bandit = LinUCB(dim=dim, alpha=a, regularization=1.0)
        res = simulate_bandit(bandit=bandit, test_df=test_df, make_context_fn=make_context_fn,
                              user_col=user_col, item_col=item_col, reward_fn=lambda u,i,row: (float(row[rating_col]) if row[item_col]==i else 0.0),
                              candidate_selector=candidate_selector, binarize=False, top_k=1, verbose=False)
        records.append({"algo":"LinUCB", "param":a, "avg_reward":res["avg_reward"], "cum_reward":res["cumulative_reward"], "rounds":res["rounds"]})

    # EpsilonGreedy sweep
    for e in epsilons:
        eps_model = EpsilonGreedy(epsilon=e)
        # simulate: EpsilonGreedy.select expects (contexts, items)
        # wrap it to match same interface: select(contexts, items)
        # We'll use simulate_bandit by passing a small wrapper object with select & update
        class EGWrapper:
            def __init__(self, eg):
                self.eg = eg
            def select(self, contexts, items=None):
                return self.eg.select(contexts, items)
            def update(self, x_or_item, reward):
                # EG expects item + reward; our simulate passes context; we pass item
                try:
                    # if x_or_item is an item id, fine
                    self.eg.update(x_or_item, reward)
                except Exception:
                    pass

        egw = EGWrapper(eps_model)
        res = simulate_bandit(bandit=egw, test_df=test_df, make_context_fn=make_context_fn,
                              user_col=user_col, item_col=item_col, reward_fn=lambda u,i,row: (float(row[rating_col]) if row[item_col]==i else 0.0),
                              candidate_selector=candidate_selector, binarize=False, top_k=1, verbose=False)
        records.append({"algo":"EpsilonGreedy", "param":e, "avg_reward":res["avg_reward"], "cum_reward":res["cumulative_reward"], "rounds":res["rounds"]})

    df = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, "bandit_param_sweep.csv")
    df.to_csv(csv_path, index=False)
    print("✅ Bandit param sweep saved:", csv_path)

    # Plot summary
    plt.figure(figsize=(8,4))
    for algo in df["algo"].unique():
        sub = df[df["algo"]==algo]
        plt.plot(sub["param"], sub["avg_reward"], marker="o", label=algo)
    plt.xlabel("Parameter (alpha for LinUCB / epsilon for EG)")
    plt.ylabel("Avg reward")
    plt.title("Bandit parameter sweep")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(charts_dir, "bandit_param_sweep.png")
    plt.savefig(plot_path)
    plt.close()
    print("📈 Bandit param sweep plot saved:", plot_path)

    return df, plot_path

# -------------------------
# Part D: Simple ablation study for context features
# -------------------------
def ablation_bandit_contexts(test_df, user_col, item_col, rating_col, results_dir,
                             pop_model=None, mf_model=None, emb=None, candidate_selector=None):
    """
    Compare LinUCB performance with:
      - full context (KG + MF if present)
      - -KG (no KG embeddings)
      - -MF (no MF embeddings)
    Returns DataFrame and saves CSV + plot.
    """
    charts_dir = os.path.join(results_dir, "charts")
    ensure_dir(charts_dir)
    unique_items = test_df[item_col].unique().tolist()

    # candidate selector
    if candidate_selector is None:
        def candidate_selector(u, row):
            try:
                return prefilter_top_k_by_popularity(pop_model, u, row, top_k=50)
            except Exception:
                return unique_items

    # variants:
    def make_ctx_full(u,i):
        return make_context_default(u, i, kg_embeddings=emb if emb is not None else None, mf_model=mf_model, item_feature_map=None, session_feats=None, concat_scale=None)

    def make_ctx_no_kg(u,i):
        return make_context_default(u, i, kg_embeddings=None, mf_model=mf_model, item_feature_map=None, session_feats=None, concat_scale=None)

    def make_ctx_no_mf(u,i):
        return make_context_default(u, i, kg_embeddings=emb if emb is not None else None, mf_model=None, item_feature_map=None, session_feats=None, concat_scale=None)

    variants = [
        ("full", make_ctx_full),
        ("no_kg", make_ctx_no_kg),
        ("no_mf", make_ctx_no_mf)
    ]

    records = []
    for name, make_ctx in variants:
        # infer dim from sample
        sample_item = unique_items[0]
        sample_user = test_df[user_col].iloc[0]
        dim = make_ctx(sample_user, sample_item).shape[0]
        bandit = LinUCB(dim=dim, alpha=1.0, regularization=1.0)
        res = simulate_bandit(bandit=bandit, test_df=test_df, make_context_fn=make_ctx,
                              user_col=user_col, item_col=item_col, reward_fn=lambda u,i,row: (float(row[rating_col]) if row[item_col]==i else 0.0),
                              candidate_selector=candidate_selector, binarize=False, top_k=1, verbose=False)
        records.append({"variant":name, "avg_reward":res["avg_reward"], "cum_reward":res["cumulative_reward"], "rounds":res["rounds"]})

    df = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, "ablation_contexts.csv")
    df.to_csv(csv_path, index=False)
    print("✅ Ablation CSV saved:", csv_path)

    # plot
    plt.figure(figsize=(6,4))
    plt.bar(df["variant"], df["avg_reward"])
    plt.ylabel("Avg reward")
    plt.title("Ablation: Context parts effect on LinUCB")
    plt.tight_layout()
    plot_path = os.path.join(charts_dir, "ablation_contexts.png")
    plt.savefig(plot_path)
    plt.close()
    print("📈 Ablation plot saved:", plot_path)
    return df, plot_path

# -------------------------
# Main run for Step14 (caller passes project variables)
# -------------------------
def run_step14(models, test_df, user_col, item_col, rating_col, results_dir,
               pop_model=None, mf_model=None, emb=None, dataset_name="Dataset"):
    ensure_dir(results_dir)
    charts_dir = os.path.join(results_dir, "charts")
    ensure_dir(charts_dir)

    print(">>> Step14: exporting final reports")
    export_final_reports(models, test_df, user_col, item_col, rating_col, results_dir, dataset_name=dataset_name)

    print(">>> Step14: building dashboard")
    dashboard_pdf = build_dashboard(results_dir)

    # Build a make_context function using existing make_context_default with scaling disabled (we use raw)
    def make_context_wrap(u,i):
        return make_context_default(u, i, kg_embeddings=emb if emb is not None else None, mf_model=mf_model, item_feature_map=None, session_feats=None, concat_scale=None)

    print(">>> Step14: bandit parameter sweep")
    sweep_df, sweep_plot = bandit_parameter_sweep(test_df=test_df, user_col=user_col, item_col=item_col, rating_col=rating_col,
                                                  results_dir=results_dir, pop_model=pop_model, mf_model=mf_model, emb=emb,
                                                  make_context_fn=make_context_wrap)

    print(">>> Step14: ablation study")
    abl_df, abl_plot = ablation_bandit_contexts(test_df=test_df, user_col=user_col, item_col=item_col, rating_col=rating_col,
                                               results_dir=results_dir, pop_model=pop_model, mf_model=mf_model, emb=emb)

    print(">>> Step14 completed. Results in:", results_dir)
    return {
        "dashboard_pdf": dashboard_pdf,
        "sweep_csv": os.path.join(results_dir, "bandit_param_sweep.csv"),
        "sweep_plot": sweep_plot,
        "ablation_csv": os.path.join(results_dir, "ablation_contexts.csv"),
        "ablation_plot": abl_plot
    }

# allow direct run for quick debug
if __name__ == "__main__":
    print("This module is intended to be imported and called from main.py")
