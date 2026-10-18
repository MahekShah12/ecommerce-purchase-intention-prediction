"""
common.py
Shared code used by BOTH dataset1_train.py and dataset2_train.py, so the
cleaning/training/plotting logic is written ONCE instead of being copy-pasted
into every dataset script.

This file does NOT load any data and does NOT train anything by itself -
it only defines reusable pieces:
  - MODELS / MODEL_COLORS   -> the 5 models used everywhere
  - make_models()           -> builds a fresh copy of all 5 models
  - metrics() / best_thr()  -> evaluation helpers
  - run_training()          -> does the split -> train -> tune -> evaluate loop
                                for ONE dataset and returns a results dict
  - save_dataset_outputs()  -> saves that dataset's metrics CSV + a pickle
                                of everything the comparison script needs
  - plot_class_balance / plot_data_split / plot_top_correlations /
    plot_metrics_bars / plot_roc_curves / plot_precision_recall /
    plot_threshold_tuning / plot_confusion_matrix / plot_feature_importance
                             -> the 9 per-dataset graphs (each its own PNG),
                                parameterised by dataset name/color so neither
                                dataset script has to redefine them
"""
import os, pickle, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier

SEED = 42
OUT = "outputs"
G = OUT + "/graphs"

# ---- 5 models, shared everywhere -------------------------------------------------
MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "XGBoost"]
MODEL_COLORS = {
    "Logistic Regression": "#4C78A8",
    "Random Forest": "#59A14F",
    "Gradient Boosting": "#E15759",
    "SVM": "#B07AA1",
    "XGBoost": "#F28E2B",
}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                      "axes.grid": True, "grid.alpha": .25, "font.size": 10, "axes.titleweight": "bold"})


def make_models(pos_weight: float):
    """Return a fresh dict of all 5 models, each handling class imbalance in its own native way."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=SEED,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED),
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, learning_rate=0.08, max_depth=5, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED, n_jobs=-1,
        ),
    }


def metrics(y, p, thr):
    pr = (p >= thr).astype(int)
    return dict(
        Accuracy=accuracy_score(y, pr), Precision=precision_score(y, pr, zero_division=0),
        Recall=recall_score(y, pr), F1=f1_score(y, pr),
        **{"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)},
    )


def best_thr(y, p):
    ts = np.arange(0.05, 0.995, 0.005)
    f = [f1_score(y, (p >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(f))]), ts, np.array(f)


# ---- training loop, run once per dataset -----------------------------------------
def run_training(ds_key: str, ds_name: str, X: pd.DataFrame, y: pd.Series):
    """
    Stratified 60/20/20 split -> train all 5 models -> tune threshold on
    validation (max F1) -> evaluate on test. SVM is slow on very large data,
    so on big datasets it trains on a stratified subsample of the training
    set only (prediction/evaluation still uses the full val/test sets).
    """
    print(f"\n===== {ds_name}  {X.shape}  positives {y.mean():.2%}")
    tv_X, te_X, tv_y, te_y = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    tr_X, va_X, tr_y, va_y = train_test_split(tv_X, tv_y, test_size=.25, stratify=tv_y, random_state=SEED)

    pos_weight = (tr_y == 0).sum() / (tr_y == 1).sum()
    models = make_models(pos_weight)

    # SVM does not scale to hundreds of thousands of rows - cap its training set
    SVM_CAP = 20000
    if len(tr_X) > SVM_CAP:
        svm_idx = tr_y.groupby(tr_y).sample(frac=SVM_CAP / len(tr_y), random_state=SEED).index
    else:
        svm_idx = tr_X.index

    r = dict(ds_name=ds_name, X=X, y=y, te_y=te_y, va_y=va_y,
             sizes=(len(tr_y), len(va_y), len(te_y)), proba={}, vproba={}, thr={}, fit={}, rows=[])

    for n, m in models.items():
        t0 = time.time()
        if n == "SVM":
            m.fit(tr_X.loc[svm_idx], tr_y.loc[svm_idx])
        else:
            m.fit(tr_X, tr_y)
        r["fit"][n] = m
        r["vproba"][n] = m.predict_proba(va_X)[:, 1]
        r["proba"][n] = m.predict_proba(te_X)[:, 1]
        r["thr"][n] = best_thr(va_y, r["vproba"][n])[0]
        for split, yy, pp in (("Validation", va_y, r["vproba"][n]), ("Test", te_y, r["proba"][n])):
            for tname, thr in (("0.50", .5), ("tuned", r["thr"][n])):
                r["rows"].append(dict(Dataset=ds_key, Model=n, Split=split, Threshold=tname,
                                       thr_value=round(thr, 2), **metrics(yy, pp, thr)))
        print(f"  {n:<20} {time.time() - t0:5.1f}s  tuned thr={r['thr'][n]:.2f}")

    vt = pd.DataFrame([x for x in r["rows"] if x["Split"] == "Validation" and x["Threshold"] == "tuned"]).set_index("Model")
    r["final"] = vt["F1"].idxmax()
    print("  FINAL MODEL:", r["final"], f"(val F1 {vt['F1'].max():.4f})")
    r["te_X"] = te_X
    return r


RESULTS_CSV = f"{OUT}/results.csv"
RESULTS_COLS = ["Dataset", "Model", "Split", "Threshold", "thr_value",
                "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]


def save_dataset_outputs(ds_key: str, r: dict):
    """Append this dataset's rows into the ONE shared outputs/results.csv (both
    datasets' model metrics end up in the same file), and save a small pickle
    with everything combined_comparison.py needs for the comparison graphs
    (NOT the fitted models themselves - just arrays/numbers)."""
    new_rows = pd.DataFrame(r["rows"])

    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
    else:
        existing = pd.DataFrame(columns=RESULTS_COLS)
    # drop any old rows for this dataset first, so re-running a script doesn't duplicate rows
    existing = existing[existing["Dataset"] != r["ds_name"]] if len(existing) else existing
    new_rows = new_rows.rename(columns={}).assign(Dataset=r["ds_name"])
    combined_df = pd.concat([existing, new_rows], ignore_index=True)[RESULTS_COLS]
    combined_df.to_csv(RESULTS_CSV, index=False)

    light = {k: v for k, v in r.items() if k not in ("fit", "X", "te_X")}
    with open(f"{OUT}/{ds_key}_results.pkl", "wb") as f:
        pickle.dump(light, f)
    print(f"  added {r['ds_name']} rows to {RESULTS_CSV}, saved {OUT}/{ds_key}_results.pkl")


def _save(fig, name):
    fig.savefig(f"{G}/{name}.png", bbox_inches="tight")
    plt.close(fig)


# ======================================================================================
# PER-DATASET GRAPHS — rewritten so EVERY chart is its own separate, single-purpose
# figure/PNG (no more 3-panels-squeezed-into-one-figure). Each function below saves
# exactly ONE file. All numbers plotted are read directly from `r` / `X` / `y` — the
# same arrays used for training — so the values in every chart match the metrics CSV.
# ======================================================================================

def _value_labels(ax, bars, values, fmt="{:.1f}", pad=None, fontsize=8.5):
    """Put the exact value on top of/after each bar so close scores stay readable."""
    if pad is None:
        pad = (max(values) - min(values) + 1e-9) * 0.015
    for b, v in zip(bars, values):
        if b.get_width() == 0 or abs(b.get_height()) > abs(b.get_width() or 1):
            ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt.format(v),
                    ha="center", va="bottom", fontsize=fontsize)


# ---- 1a. Class balance --------------------------------------------------------------
def plot_class_balance(ds_key, ds_name, color, y):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    vc = y.value_counts().sort_index()
    bars = ax.bar(["No purchase", "Purchase"], vc.values, color=["#B0B7C3", color], width=.55)
    for rect, v in zip(bars, vc.values):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:,}\n({v/len(y):.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set(title=f"{ds_name}\nClass balance (target distribution)", ylabel="sessions / users")
    ax.set_ylim(0, vc.max() * 1.22)
    plt.tight_layout(); _save(fig, f"{ds_key}_1a_class_balance")


# ---- 1b. Train/val/test split --------------------------------------------------------
def plot_data_split(ds_key, ds_name, color, sizes):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    s1, s2, s3 = sizes
    total = s1 + s2 + s3
    ax.pie([s1, s2, s3], labels=[f"Train\n{s1:,} ({s1/total:.0%})", f"Val\n{s2:,} ({s2/total:.0%})",
                                  f"Test\n{s3:,} ({s3/total:.0%})"],
           colors=[color, "#F4A261", "#264653"], autopct=None,
           wedgeprops=dict(width=.45, edgecolor="w"), textprops=dict(fontsize=10))
    ax.set_title(f"{ds_name}\nStratified 60 / 20 / 20 split  (total {total:,} rows)")
    plt.tight_layout(); _save(fig, f"{ds_key}_1b_data_split")


# ---- 1c. Top correlated features -----------------------------------------------------
def plot_top_correlations(ds_key, ds_name, color, X, y):
    corr = pd.concat([X.select_dtypes("number").loc[:, X.nunique() > 1], y.rename("target")], axis=1) \
             .corr()["target"].drop("target")
    top = corr.reindex(corr.abs().sort_values(ascending=False).index)[:12].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    bars = ax.barh(top.index, top.values, color=[color if v > 0 else "#9AA0A6" for v in top.values])
    for b, v in zip(bars, top.values):
        ax.text(v + (0.012 if v >= 0 else -0.012), b.get_y() + b.get_height() / 2, f"{v:.2f}",
                ha="left" if v >= 0 else "right", va="center", fontsize=8.5)
    ax.axvline(0, color="#444", linewidth=.8)
    ax.set(title=f"{ds_name}\nTop-12 features correlated with purchase", xlabel="Pearson r")
    xmax = max(abs(top.min()), abs(top.max())) * 1.25
    ax.set_xlim(-xmax, xmax)
    plt.tight_layout(); _save(fig, f"{ds_key}_1c_top_correlations")


# ---- 2a. Metrics bar chart, all 5 models, zoomed y-axis so close scores are visible --
def plot_metrics_bars(ds_key, ds_name, r):
    mets = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    T = pd.DataFrame([x for x in r["rows"] if x["Split"] == "Test" and x["Threshold"] == "tuned"]).set_index("Model")
    vals = {n: [T.loc[n, m] * 100 for m in mets] for n in MODELS}
    all_vals = [v for row in vals.values() for v in row]
    lo = max(0, min(all_vals) - 8)   # zoom the y-axis to where the bars actually differ

    fig, ax = plt.subplots(figsize=(11, 5.6))
    w = .16
    for k, n in enumerate(MODELS):
        bars = ax.bar(np.arange(5) + (k - 2) * w, vals[n], w, label=n, color=MODEL_COLORS[n])
        for bb, vv in zip(bars, vals[n]):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.3, f"{vv:.1f}", ha="center",
                    fontsize=7.3, rotation=90 if lo > 50 else 0)
    ax.set_xticks(range(5)); ax.set_xticklabels(mets, fontsize=11)
    ax.set(ylim=(lo, 101.5), ylabel="%", title=f"{ds_name}\nTest-set metrics, all 5 models (validation-tuned threshold)")
    ax.legend(loc="lower left", ncol=3, fontsize=8, framealpha=.9)
    plt.tight_layout(); _save(fig, f"{ds_key}_2a_metrics_bars")


# ---- 2b. ROC curves, with a zoomed inset on the top-left corner ---------------------
def plot_roc_curves(ds_key, ds_name, r):
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    for n in MODELS:
        f, t, _ = roc_curve(r["te_y"], r["proba"][n])
        ax.plot(f, t, lw=2.2, color=MODEL_COLORS[n],
                label=f"{n} (AUC {roc_auc_score(r['te_y'], r['proba'][n]):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1)
    ax.set(title=f"{ds_name}\nROC curves — all 5 models (test)",
           xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=8.3)

    # zoomed inset, since all curves hug the top-left corner and overlap there
    axins = ax.inset_axes([0.42, 0.12, 0.46, 0.46])
    for n in MODELS:
        f, t, _ = roc_curve(r["te_y"], r["proba"][n])
        axins.plot(f, t, lw=2.0, color=MODEL_COLORS[n])
    axins.set_xlim(0, 0.15); axins.set_ylim(0.85, 1.005)
    axins.set_title("zoom: top-left corner", fontsize=8)
    axins.tick_params(labelsize=7)
    ax.indicate_inset_zoom(axins, edgecolor="gray")
    plt.tight_layout(); _save(fig, f"{ds_key}_2b_roc_curves")


# ---- 3a. Precision-Recall curves, all 5 models ---------------------------------------
def plot_precision_recall(ds_key, ds_name, r):
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for n in MODELS:
        p, rc, _ = precision_recall_curve(r["te_y"], r["proba"][n])
        ax.plot(rc, p, lw=2.1, color=MODEL_COLORS[n],
                label=f"{n} (AP {average_precision_score(r['te_y'], r['proba'][n]):.3f})")
    ax.set(title=f"{ds_name}\nPrecision-Recall curves — all 5 models (test)",
           xlabel="Recall", ylabel="Precision", ylim=(0, 1.03))
    ax.legend(fontsize=8.3, loc="lower left")
    plt.tight_layout(); _save(fig, f"{ds_key}_3a_precision_recall")


# ---- 3b. Threshold tuning curve for the FINAL model only ------------------------------
def plot_threshold_tuning(ds_key, ds_name, color, r):
    fm = r["final"]
    _, ts, f1s = best_thr(r["va_y"], r["vproba"][fm])
    pp, rr = [], []
    for t in ts:
        pr_ = (r["vproba"][fm] >= t).astype(int)
        pp.append(precision_score(r["va_y"], pr_, zero_division=0)); rr.append(recall_score(r["va_y"], pr_))
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.plot(ts, pp, label="Precision", color="#4C78A8", lw=2)
    ax.plot(ts, rr, label="Recall", color="#F28E2B", lw=2)
    ax.plot(ts, f1s, label="F1", color=color, lw=2.8)
    chosen = r["thr"][fm]
    ax.axvline(chosen, ls="--", c="k", lw=1.2)
    ax.scatter([chosen], [max(f1s)], color=color, zorder=5, s=45, edgecolor="k", linewidth=.6)
    ax.text(chosen + .015, 0.04, f"chosen threshold = {chosen:.2f}\n(max validation F1 = {max(f1s):.3f})", fontsize=9)
    ax.set(title=f"{ds_name}\nThreshold tuning on validation — final model: {fm}",
           xlabel="decision threshold", ylabel="score", ylim=(0, 1.05))
    ax.legend(fontsize=9)
    plt.tight_layout(); _save(fig, f"{ds_key}_3b_threshold_tuning")


# ---- 3c. Confusion matrix for the FINAL model, standalone (fixes the old aspect bug) --
def plot_confusion_matrix(ds_key, ds_name, r):
    fm = r["final"]
    cm = confusion_matrix(r["te_y"], (r["proba"][fm] >= r["thr"][fm]).astype(int))
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1%})", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="w" if cm_pct[i, j] > 0.5 else "k")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No purchase", "Purchase"]); ax.set_yticklabels(["No purchase", "Purchase"])
    ax.set(title=f"{ds_name}\nConfusion matrix (test) — final model: {fm}\nthreshold = {r['thr'][fm]:.2f}",
           xlabel="Predicted", ylabel="Actual")
    fig.colorbar(im, ax=ax, label="row-normalised %", fraction=0.046, pad=0.04)
    plt.tight_layout(); _save(fig, f"{ds_key}_3c_confusion_matrix")


# ---- 4. Feature importance — value labels so the plot is meaningful even when one
#         feature dominates the others --------------------------------------------------
def plot_feature_importance(ds_key, ds_name, color, r):
    fm = r["final"]; te_X = r["te_X"]
    idx = np.random.RandomState(SEED).choice(len(te_X), min(len(te_X), 6000), replace=False)
    pi = permutation_importance(r["fit"][fm], te_X.iloc[idx], r["te_y"].iloc[idx],
                                 scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, te_X.columns).sort_values(ascending=False)[:12].iloc[::-1]
    imp_plot = imp.clip(lower=0)   # permutation importance can be slightly negative from noise

    fig, ax = plt.subplots(figsize=(9, 6.4))
    bars = ax.barh(imp_plot.index, imp_plot.values, color=color)
    for b, v in zip(bars, imp_plot.values):
        ax.text(v + imp_plot.max() * 0.015, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=8.5)
    ax.set(title=f"{ds_name}\nTop-12 features ({fm}) — permutation importance\n(mean drop in ROC-AUC when the feature is shuffled)",
           xlabel="importance (drop in ROC-AUC)")
    ax.set_xlim(0, imp_plot.max() * 1.22)
    plt.tight_layout(); _save(fig, f"{ds_key}_4_feature_importance")
    return imp