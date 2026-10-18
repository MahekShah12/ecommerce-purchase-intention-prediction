"""
combined_comparison.py
ONLY compares Dataset 1 vs Dataset 2 results. Does NOT load raw data, does
NOT clean/engineer features, does NOT train any model.

Run this LAST, after both:
    python dataset1_train.py
    python dataset2_train.py
have each produced outputs/ds1_results.pkl and outputs/ds2_results.pkl.

This script just loads those two pickles and draws the cross-dataset,
cross-model comparison graphs (which model wins overall, heatmaps of every
model x every dataset x metric, etc.) plus one merged metrics CSV. Every
chart is its own separate PNG.
"""
import os, pickle, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import common
from common import MODELS, MODEL_COLORS, OUT, G

DATASET_COLORS = {"Dataset 1": "#2A9D8F", "Dataset 2": "#E76F51"}
os.makedirs(G, exist_ok=True)


def load_results():
    missing = [p for p in ("ds1", "ds2") if not os.path.exists(f"{OUT}/{p}_results.pkl")]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} results. Run dataset1_train.py and dataset2_train.py first."
        )
    with open(f"{OUT}/ds1_results.pkl", "rb") as f:
        r1 = pickle.load(f)
    with open(f"{OUT}/ds2_results.pkl", "rb") as f:
        r2 = pickle.load(f)
    return {"Dataset 1": r1, "Dataset 2": r2}


if __name__ == "__main__":
    R = load_results()
    d1, d2 = "Dataset 1", "Dataset 2"
    NAMES = {d: R[d]["ds_name"] for d in R}
    mets = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    # ---- read the ONE shared outputs/results.csv that both training scripts appended to ----
    results_path = f"{OUT}/results.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found. Run dataset1_train.py and dataset2_train.py first.")
    M = pd.read_csv(results_path)
    # results.csv stores the full descriptive dataset name; map it back to the short
    # "Dataset 1" / "Dataset 2" keys used everywhere below in this script
    name_to_key = {R[d]["ds_name"]: d for d in R}
    M["Dataset"] = M["Dataset"].map(name_to_key)
    if M["Dataset"].isna().any():
        raise ValueError("results.csv has dataset names that don't match the loaded pickles - re-run the training scripts.")

    T = M[(M.Split == "Test") & (M.Threshold == "tuned")].set_index(["Dataset", "Model"])
    V = M[(M.Split == "Validation") & (M.Threshold == "tuned")].set_index(["Dataset", "Model"])

    final_rows = [
        dict(Dataset=NAMES[d], FinalModel=R[d]["final"], Threshold=R[d]["thr"][R[d]["final"]],
             **T.loc[(d, R[d]["final"]), ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]].to_dict())
        for d in R
    ]
    pd.DataFrame(final_rows).round(4).to_csv(f"{OUT}/final_models_test_metrics.csv", index=False)
    print(pd.DataFrame(final_rows).round(4).to_string())

    # ==================================================================================
    # Every comparison chart below is now its OWN figure/PNG (no more 2-4 panels
    # squeezed into one wide figure), and bar charts use a zoomed y-axis + exact value
    # labels so close scores between models/datasets are still visible.
    # ==================================================================================

    # ---- C1a: final model on each dataset — metrics bar chart ----
    fig, ax = plt.subplots(figsize=(9.5, 6))
    w = .36
    all_v = []
    for k, d in enumerate([d1, d2]):
        v = [T.loc[(d, R[d]["final"]), m] * 100 for m in mets]
        all_v += v
        bars = ax.bar(np.arange(5) + (k - .5) * w, v, w, color=DATASET_COLORS[d], label=f"{d}: {R[d]['final']}")
        for bb, vv in zip(bars, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + .8, f"{vv:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(5)); ax.set_xticklabels(mets, fontsize=11)
    lo = max(0, min(all_v) - 10)
    ax.set(ylim=(lo, 103), ylabel="%", title="Final model on each dataset (held-out test)")
    ax.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout(); fig.savefig(f"{G}/C1a_final_models_metrics.png", bbox_inches="tight"); plt.close(fig)

    # ---- C1b: ROC of final models ----
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    for d in (d1, d2):
        fm = R[d]["final"]
        f, t, _ = roc_curve(R[d]["te_y"], R[d]["proba"][fm])
        ax.plot(f, t, lw=2.6, color=DATASET_COLORS[d], label=f"{d} - {fm} (AUC {roc_auc_score(R[d]['te_y'], R[d]['proba'][fm]):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1)
    ax.set(title="ROC curves of each dataset's final model", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout(); fig.savefig(f"{G}/C1b_final_models_roc.png", bbox_inches="tight"); plt.close(fig)

    # ---- C2: heatmap, every model x every dataset - ONE separate heatmap per metric ----
    for m in ["F1", "Precision", "Recall", "ROC-AUC"]:
        hm = np.array([[T.loc[(d, n), m] for d in (d1, d2)] for n in MODELS])
        fig, ax = plt.subplots(figsize=(5.4, 5.6))
        ax.grid(False)
        im = ax.imshow(hm, cmap="YlGnBu", vmin=.4, vmax=1)
        for i in range(len(MODELS)):
            for j in range(2):
                ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center", fontweight="bold",
                        color="w" if hm[i, j] > .78 else "k")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Dataset 1", "Dataset 2"])
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS)
        ax.set_title(f"{m} — every model x every dataset\n(test, tuned threshold)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        metric_tag = m.replace("-", "").lower()
        fig.savefig(f"{G}/C2_heatmap_{metric_tag}.png", bbox_inches="tight"); plt.close(fig)

    # ---- C3a: overall ranking across both datasets — bar chart ----
    sc = T.reset_index()
    sc["rank_F1"] = sc.groupby("Dataset")["F1"].rank(ascending=False)
    sc["rank_AUC"] = sc.groupby("Dataset")["ROC-AUC"].rank(ascending=False)
    ov = sc.groupby("Model").agg(
        mean_F1=("F1", "mean"), mean_ROC_AUC=("ROC-AUC", "mean"), mean_PR_AUC=("PR-AUC", "mean"),
        mean_Precision=("Precision", "mean"), mean_Recall=("Recall", "mean"),
        avg_rank_F1=("rank_F1", "mean"), avg_rank_AUC=("rank_AUC", "mean"),
    ).loc[MODELS]
    ov["wins_F1"] = [int(sum(R[d]["final"] == n for d in R)) for n in MODELS]
    ov["overall_score"] = (ov.mean_F1 + ov.mean_ROC_AUC + ov.mean_PR_AUC) / 3
    ov = ov.sort_values("overall_score", ascending=False)
    ov.round(4).to_csv(f"{OUT}/overall_model_ranking.csv")
    print(ov.round(4).to_string())
    overall_best = ov.index[0]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    x = np.arange(len(ov))
    vals_all = []
    for k, (m, lab) in enumerate([("mean_F1", "Mean F1"), ("mean_ROC_AUC", "Mean ROC-AUC"), ("mean_PR_AUC", "Mean PR-AUC")]):
        v = ov[m] * 100
        vals_all += list(v)
        bars = ax.bar(x + (k - 1) * .26, v, .26, label=lab, color=["#264653", "#2A9D8F", "#E9C46A"][k])
        for bb, vv in zip(bars, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + .8, f"{vv:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(ov.index, rotation=20, ha="right", fontsize=10)
    lo = max(0, min(vals_all) - 10)
    ax.set(ylim=(lo, 103), ylabel="% (average of both datasets)", title=f"Overall comparison across both datasets — best: {overall_best}")
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    plt.tight_layout(); fig.savefig(f"{G}/C3a_overall_ranking.png", bbox_inches="tight"); plt.close(fig)

    # ---- C3b: F1 by dataset, one line per model ----
    fig, ax = plt.subplots(figsize=(7, 6.2))
    for n in MODELS:
        ax.plot([d1, d2], [T.loc[(d, n), "F1"] for d in (d1, d2)], marker="o", lw=2.4, ms=9,
                color=MODEL_COLORS[n], label=n)
        for d in (d1, d2):
            v = T.loc[(d, n), "F1"]
            ax.annotate(f"{v:.3f}", (d, v), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set(title="F1 by dataset — every model", ylabel="F1")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(); fig.savefig(f"{G}/C3b_f1_by_dataset.png", bbox_inches="tight"); plt.close(fig)

    # ---- C9: validation vs test F1 — one file PER DATASET (generalisation check) ----
    for d in (d1, d2):
        fig, ax = plt.subplots(figsize=(8.6, 6))
        xx = np.arange(len(MODELS))
        for k, (lab, df_, c) in enumerate((("Validation", V, "#9AA0A6"), ("Test", T, DATASET_COLORS[d]))):
            v = [df_.loc[(d, n), "F1"] for n in MODELS]
            b = ax.bar(xx + (k - .5) * .36, v, .36, label=lab, color=c)
            for bb, vv in zip(b, v):
                ax.text(bb.get_x() + bb.get_width() / 2, vv + .01, f"{vv:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(xx); ax.set_xticklabels(MODELS, rotation=20, ha="right", fontsize=10)
        ax.set(title=f"{NAMES[d]}\nValidation vs Test F1 (tuned threshold) — generalisation check",
               ylim=(0, 1.1), ylabel="F1")
        ax.legend(loc="lower right", fontsize=9.5)
        plt.tight_layout()
        tag = "ds1" if d == d1 else "ds2"
        fig.savefig(f"{G}/C9_{tag}_validation_vs_test_f1.png", bbox_inches="tight"); plt.close(fig)

    json.dump(
        {"overall_best": overall_best, "final": {d: R[d]["final"] for d in R}, "thr": {d: R[d]["thr"] for d in R}},
        open(f"{OUT}/summary.json", "w"), indent=1,
    )
    print("\nCOMPARISON DONE. Overall best model:", overall_best)