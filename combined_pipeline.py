import warnings, time, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")
SEED = 42
OUT = "outputs"; G = OUT + "/graphs"
MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
MC = {"Logistic Regression": "#4C78A8", "Random Forest": "#59A14F", "Gradient Boosting": "#E15759"}
DC = {"Dataset 1": "#2A9D8F", "Dataset 2": "#E76F51"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10, "axes.titleweight": "bold"})
figs = []
def save(fig, name):
    fig.savefig(f"{G}/{name}.png", bbox_inches="tight"); figs.append(fig)

# ---------------- load + feature engineering ----------------
def load1():
    df = pd.read_csv("data/dataset1_clickstream_training_sample.csv")
    df = df.drop_duplicates(subset="UserID"); y = df["ordered"]
    X = df.drop(columns=["UserID", "ordered"])
    X["total_clicks"] = X[["basket_icon_click","basket_add_list","basket_add_detail","sort_by","image_picker",
        "account_page_click","promo_banner_click","detail_wishlist_add","list_size_dropdown","closed_minibasket_click"]].sum(axis=1)
    X["info_checks"] = X[["checked_delivery_detail","checked_returns_detail","saw_sizecharts","saw_delivery"]].sum(axis=1)
    return X, y
def load2():
    df = pd.read_csv("data/dataset2_online_shoppers_intention.csv")
    n0 = len(df); df = df.drop_duplicates()
    y = df["Revenue"].astype(str).str.upper().eq("TRUE").astype(int)
    X = df.drop(columns="Revenue").copy(); X["Weekend"] = X["Weekend"].astype(str).str.upper().eq("TRUE").astype(int)
    X["TotalPages"] = X.Administrative + X.Informational + X.ProductRelated
    X["TotalDuration"] = X.Administrative_Duration + X.Informational_Duration + X.ProductRelated_Duration
    X["AvgTimePerPage"] = (X.TotalDuration / X.TotalPages.replace(0, np.nan)).fillna(0)
    X["ExitBounceDiff"] = X.ExitRates - X.BounceRates
    X["PageValuePerProductPage"] = (X.PageValues / X.ProductRelated.replace(0, np.nan)).fillna(0)
    X = pd.get_dummies(X, columns=["Month","VisitorType","OperatingSystems","Browser","Region","TrafficType"], dtype=int)
    print(f"D2 duplicates removed: {n0-len(df)}")
    return X, y
DATA = {"Dataset 1": load1(), "Dataset 2": load2()}
NAMES = {"Dataset 1": "Dataset 1 (Clickstream, 'ordered')", "Dataset 2": "Dataset 2 (UCI Online Shoppers, 'Revenue')"}

def make_models(y_tr):
    pw = (y_tr == 0).sum() / (y_tr == 1).sum()
    sw = None
    return {
      "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
      "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
      "Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=SEED),
    }
def metrics(y, p, thr):
    pr = (p >= thr).astype(int)
    return dict(Accuracy=accuracy_score(y, pr), Precision=precision_score(y, pr, zero_division=0), Recall=recall_score(y, pr),
                F1=f1_score(y, pr), **{"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)})
def best_thr(y, p):
    ts = np.arange(0.05, 0.995, 0.005); f = [f1_score(y, (p >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(f))]), ts, np.array(f)

R = {}; rows = []
for ds, (X, y) in DATA.items():
    print("\n=====", NAMES[ds], X.shape, f"positives {y.mean():.2%}")
    tv_X, te_X, tv_y, te_y = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    tr_X, va_X, tr_y, va_y = train_test_split(tv_X, tv_y, test_size=.25, stratify=tv_y, random_state=SEED)
    models = make_models(tr_y); r = dict(X=X, y=y, te_y=te_y, va_y=va_y, sizes=(len(tr_y), len(va_y), len(te_y)), proba={}, vproba={}, thr={}, fit={})
    for n, m in models.items():
        t0 = time.time(); m.fit(tr_X, tr_y); r["fit"][n] = m
        r["vproba"][n] = m.predict_proba(va_X)[:, 1]; r["proba"][n] = m.predict_proba(te_X)[:, 1]
        r["thr"][n] = best_thr(va_y, r["vproba"][n])[0]
        for split, yy, pp in (("Validation", va_y, r["vproba"][n]), ("Test", te_y, r["proba"][n])):
            for tname, thr in (("0.50", .5), ("tuned", r["thr"][n])):
                rows.append(dict(Dataset=ds, Model=n, Split=split, Threshold=tname, thr_value=round(thr, 2), **metrics(yy, pp, thr)))
        print(f"  {n:<20} {time.time()-t0:5.1f}s  tuned thr={r['thr'][n]:.2f}")
    vt = pd.DataFrame([x for x in rows if x["Dataset"] == ds and x["Split"] == "Validation" and x["Threshold"] == "tuned"]).set_index("Model")
    r["final"] = vt["F1"].idxmax(); print("  FINAL MODEL:", r["final"], f"(val F1 {vt['F1'].max():.4f})")
    r["te_X"] = te_X; R[ds] = r
M = pd.DataFrame(rows); M.to_csv(OUT + "/all_model_metrics.csv", index=False)
T = M[(M.Split == "Test") & (M.Threshold == "tuned")].set_index(["Dataset", "Model"])
final_rows = [dict(Dataset=NAMES[d], FinalModel=R[d]["final"], Threshold=R[d]["thr"][R[d]["final"]], **T.loc[(d, R[d]["final"]), ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]].to_dict()) for d in R]
pd.DataFrame(final_rows).round(4).to_csv(OUT + "/final_models_test_metrics.csv", index=False)
print(pd.DataFrame(final_rows).round(4).to_string())

# ---------------- per-dataset graphs ----------------
mets = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
for ds, r in R.items():
    X, y = r["X"], r["y"]; col = DC[ds]; fm = r["final"]
    # 1 overview
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6)); fig.suptitle(NAMES[ds] + " - data overview", fontsize=14, fontweight="bold")
    vc = y.value_counts().sort_index(); b = ax[0].bar(["No purchase", "Purchase"], vc.values, color=["#B0B7C3", col])
    for rect, v in zip(b, vc.values): ax[0].text(rect.get_x()+rect.get_width()/2, v, f"{v:,}\n({v/len(y):.1%})", ha="center", va="bottom")
    ax[0].set(title="Class balance", ylabel="sessions / users"); ax[0].set_ylim(0, vc.max()*1.2)
    s1, s2, s3 = r["sizes"]; ax[1].pie([s1, s2, s3], labels=[f"Train {s1:,}", f"Val {s2:,}", f"Test {s3:,}"], colors=[col, "#F4A261", "#264653"], autopct="%1.0f%%", wedgeprops=dict(width=.45, edgecolor="w")); ax[1].set_title("Stratified 60/20/20 split")
    base = X.copy(); 
    corr = pd.concat([base.select_dtypes("number").loc[:, base.nunique() > 1], y.rename("target")], axis=1).corr()["target"].drop("target")
    top = corr.reindex(corr.abs().sort_values(ascending=False).index)[:10].iloc[::-1]
    ax[2].barh(top.index, top.values, color=[col if v > 0 else "#9AA0A6" for v in top.values]); ax[2].set(title="Top-10 correlation with purchase", xlabel="Pearson r")
    plt.tight_layout(); save(fig, f"{ds[-1]}_1_overview")
    # 2 all-models test bars
    fig, ax = plt.subplots(1, 2, figsize=(17, 5)); fig.suptitle(NAMES[ds] + " - evaluation of all models (held-out test)", fontsize=14, fontweight="bold")
    w = .26
    for k, n in enumerate(MODELS):
        v = [T.loc[(ds, n), m] * 100 for m in mets]
        bars = ax[0].bar(np.arange(5) + (k-1)*w, v, w, label=n, color=MC[n])
        for bb, vv in zip(bars, v): ax[0].text(bb.get_x()+bb.get_width()/2, vv+.5, f"{vv:.1f}", ha="center", fontsize=7.5)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(mets); ax[0].set(ylim=(0, 112), ylabel="%", title="Metrics at validation-tuned threshold"); ax[0].legend(loc="lower left", ncol=3, fontsize=8)
    for n in MODELS:
        f, t, _ = roc_curve(r["te_y"], r["proba"][n]); ax[1].plot(f, t, lw=2.2, color=MC[n], label=f"{n} (AUC {roc_auc_score(r['te_y'], r['proba'][n]):.3f})")
    ax[1].plot([0, 1], [0, 1], "--", c="gray"); ax[1].set(title="ROC curves (test)", xlabel="False positive rate", ylabel="True positive rate"); ax[1].legend(loc="lower right")
    plt.tight_layout(); save(fig, f"{ds[-1]}_2_model_evaluation")
    # 3 PR + threshold + CM of final
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8)); fig.suptitle(f"{NAMES[ds]} - final model: {fm}", fontsize=14, fontweight="bold")
    for n in MODELS:
        p, rc, _ = precision_recall_curve(r["te_y"], r["proba"][n]); ax[0].plot(rc, p, lw=2.2, color=MC[n], label=f"{n} (AP {average_precision_score(r['te_y'], r['proba'][n]):.3f})")
    ax[0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision"); ax[0].legend(fontsize=8)
    _, ts, f1s = best_thr(r["va_y"], r["vproba"][fm]); pp, rr = [], []
    for t in ts: pr_ = (r["vproba"][fm] >= t).astype(int); pp.append(precision_score(r["va_y"], pr_, zero_division=0)); rr.append(recall_score(r["va_y"], pr_))
    ax[1].plot(ts, pp, label="Precision", color="#4C78A8"); ax[1].plot(ts, rr, label="Recall", color="#F28E2B"); ax[1].plot(ts, f1s, label="F1", color=col, lw=2.6)
    ax[1].axvline(r["thr"][fm], ls="--", c="k"); ax[1].text(r["thr"][fm]+.01, .05, f"chosen = {r['thr'][fm]:.2f}"); ax[1].set(title="Threshold tuning (validation)", xlabel="threshold"); ax[1].legend()
    cm = confusion_matrix(r["te_y"], (r["proba"][fm] >= r["thr"][fm]).astype(int)); ax[2].grid(False)
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax[2].text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=15, fontweight="bold", color="w" if cm[i,j] > cm.max()/2 else "k")
    ax[2].set_xticks([0,1]); ax[2].set_yticks([0,1]); ax[2].set_xticklabels(["No purchase","Purchase"]); ax[2].set_yticklabels(["No purchase","Purchase"]); ax[2].set(title="Confusion matrix (test)", xlabel="Predicted", ylabel="Actual")
    plt.tight_layout(); save(fig, f"{ds[-1]}_3_final_model")
    # 4 feature importance (permutation, final model, subsample of test)
    te_X = r["te_X"]; idx = np.random.RandomState(SEED).choice(len(te_X), min(len(te_X), 6000), replace=False)
    pi = permutation_importance(r["fit"][fm], te_X.iloc[idx], r["te_y"].iloc[idx], scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, te_X.columns).sort_values(ascending=False)[:12].iloc[::-1]; r["imp"] = imp
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.barh(imp.index, imp.values, color=col); ax.set(title=f"{NAMES[ds]}\nTop-12 features ({fm}, permutation importance, drop in ROC-AUC)", xlabel="importance")
    plt.tight_layout(); save(fig, f"{ds[-1]}_4_feature_importance")

# ---------------- comparison graphs ----------------
d1, d2 = "Dataset 1", "Dataset 2"
# C1 final-model metrics side by side
fig, ax = plt.subplots(1, 2, figsize=(17, 5.2)); fig.suptitle("Final model on each dataset (held-out test)", fontsize=14, fontweight="bold")
w = .36
for k, d in enumerate([d1, d2]):
    v = [T.loc[(d, R[d]["final"]), m]*100 for m in mets]; bars = ax[0].bar(np.arange(5)+(k-.5)*w, v, w, color=DC[d], label=f"{d}: {R[d]['final']}")
    for bb, vv in zip(bars, v): ax[0].text(bb.get_x()+bb.get_width()/2, vv+.6, f"{vv:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(mets); ax[0].set(ylim=(0, 112), ylabel="%", title="Metrics"); ax[0].legend(loc="upper center", bbox_to_anchor=(.5,-.08), ncol=2)
for d in (d1, d2):
    fm = R[d]["final"]; f, t, _ = roc_curve(R[d]["te_y"], R[d]["proba"][fm]); ax[1].plot(f, t, lw=2.6, color=DC[d], label=f"{d} - {fm} (AUC {roc_auc_score(R[d]['te_y'], R[d]['proba'][fm]):.3f})")
ax[1].plot([0,1],[0,1],"--",c="gray"); ax[1].set(title="ROC of final models", xlabel="FPR", ylabel="TPR"); ax[1].legend(loc="lower right")
plt.tight_layout(); save(fig, "C1_final_models_comparison")
# C2 heatmaps model x dataset for F1 / ROC-AUC / Recall / Precision
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8)); fig.suptitle("Every model on both datasets (test, tuned threshold)", fontsize=14, fontweight="bold")
for a, m in zip(ax, ["F1", "Precision", "Recall", "ROC-AUC"]):
    hm = np.array([[T.loc[(d, n), m] for d in (d1, d2)] for n in MODELS]); a.grid(False); a.imshow(hm, cmap="YlGnBu", vmin=.4, vmax=1)
    for i in range(3):
        for j in range(2): a.text(j, i, f"{hm[i,j]:.3f}", ha="center", va="center", fontweight="bold", color="w" if hm[i,j] > .78 else "k")
    a.set_xticks([0,1]); a.set_xticklabels(["Dataset 1","Dataset 2"]); a.set_yticks(range(3)); a.set_yticklabels(MODELS if m == "F1" else [""]*3); a.set_title(m)
plt.tight_layout(); save(fig, "C2_model_by_dataset_heatmap")
# C3 overall ranking
sc = T.reset_index(); sc["rank_F1"] = sc.groupby("Dataset")["F1"].rank(ascending=False); sc["rank_AUC"] = sc.groupby("Dataset")["ROC-AUC"].rank(ascending=False)
ov = sc.groupby("Model").agg(mean_F1=("F1","mean"), mean_ROC_AUC=("ROC-AUC","mean"), mean_PR_AUC=("PR-AUC","mean"), mean_Precision=("Precision","mean"), mean_Recall=("Recall","mean"), avg_rank_F1=("rank_F1","mean"), avg_rank_AUC=("rank_AUC","mean")).loc[MODELS]
ov["wins_F1"] = [int(sum(R[d]["final"] == n for d in R)) for n in MODELS]
ov["overall_score"] = (ov.mean_F1 + ov.mean_ROC_AUC + ov.mean_PR_AUC)/3; ov = ov.sort_values("overall_score", ascending=False); ov.round(4).to_csv(OUT + "/overall_model_ranking.csv")
print(ov.round(4).to_string()); overall_best = ov.index[0]
fig, ax = plt.subplots(1, 2, figsize=(17, 4.8)); fig.suptitle(f"Overall comparison across both datasets - best: {overall_best}", fontsize=14, fontweight="bold")
x = np.arange(len(ov)); 
for k, (m, lab) in enumerate([("mean_F1","Mean F1"),("mean_ROC_AUC","Mean ROC-AUC"),("mean_PR_AUC","Mean PR-AUC")]):
    bars = ax[0].bar(x+(k-1)*.26, ov[m]*100, .26, label=lab, color=["#264653","#2A9D8F","#E9C46A"][k])
    for bb, vv in zip(bars, ov[m]*100): ax[0].text(bb.get_x()+bb.get_width()/2, vv+.5, f"{vv:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(ov.index); ax[0].set(ylim=(0,112), ylabel="% (average of both datasets)", title="Average score"); ax[0].legend(loc="lower left", ncol=3)
for n in MODELS:
    ax[1].plot(["Dataset 1","Dataset 2"], [T.loc[(d, n), "F1"] for d in (d1, d2)], marker="o", lw=2.6, ms=9, color=MC[n], label=n)
ax[1].set(title="F1 by dataset", ylabel="F1"); ax[1].legend()
plt.tight_layout(); save(fig, "C3_overall_best_model")
# C4 PR overlay of final models + class balance contrast
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for d in (d1, d2):
    fm = R[d]["final"]; p, rc, _ = precision_recall_curve(R[d]["te_y"], R[d]["proba"][fm]); ax[0].plot(rc, p, lw=2.6, color=DC[d], label=f"{d} - {fm}")
    ax[0].axhline(R[d]["te_y"].mean(), ls=":", color=DC[d]); 
ax[0].set(title="Precision-Recall of final models (dotted = random baseline)", xlabel="Recall", ylabel="Precision"); ax[0].legend()
ax[1].bar(["Dataset 1","Dataset 2"], [R[d]["y"].mean()*100 for d in (d1, d2)], color=[DC[d1], DC[d2]]); 
for i, d in enumerate((d1, d2)): ax[1].text(i, R[d]["y"].mean()*100+.3, f"{R[d]['y'].mean():.1%}\n(n={len(R[d]['y']):,})", ha="center")
ax[1].set(title="Purchase rate (class imbalance)", ylabel="% purchases", ylim=(0, 20)); plt.tight_layout(); save(fig, "C4_pr_and_imbalance")


# ---------------- dedicated ROC charts ----------------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6)); fig.suptitle("ROC curves - all models on both datasets (held-out test)", fontsize=14, fontweight="bold")
for a, d in zip(ax[:2], (d1, d2)):
    for n in MODELS:
        f, t, _ = roc_curve(R[d]["te_y"], R[d]["proba"][n]); a.plot(f, t, lw=2.4, color=MC[n], label=f"{n} (AUC {roc_auc_score(R[d]['te_y'], R[d]['proba'][n]):.4f})")
    a.plot([0,1],[0,1],"--",c="gray",label="Random (AUC 0.5)"); a.fill_between(*roc_curve(R[d]["te_y"], R[d]["proba"][R[d]["final"]])[:2], alpha=.08, color=DC[d])
    a.set(title=NAMES[d], xlabel="False positive rate", ylabel="True positive rate"); a.legend(loc="lower right", fontsize=8.5)
ls = {"Logistic Regression": "-", "Random Forest": "--", "Gradient Boosting": ":"}
for d in (d1, d2):
    for n in MODELS:
        f, t, _ = roc_curve(R[d]["te_y"], R[d]["proba"][n]); ax[2].plot(f, t, lw=2.2, ls=ls[n], color=DC[d], label=f"{d} - {n} ({roc_auc_score(R[d]['te_y'], R[d]['proba'][n]):.3f})")
ax[2].plot([0,1],[0,1],"--",c="gray"); ax[2].set(title="All 6 model-dataset ROC curves", xlabel="False positive rate", ylabel="True positive rate"); ax[2].legend(loc="lower right", fontsize=7.5)
plt.tight_layout(); save(fig, "C5_roc_curves_all")
for d in (d1, d2):
    fig, ax = plt.subplots(figsize=(7, 6))
    for n in MODELS:
        f, t, _ = roc_curve(R[d]["te_y"], R[d]["proba"][n]); ax.plot(f, t, lw=2.6, color=MC[n], label=f"{n} (AUC {roc_auc_score(R[d]['te_y'], R[d]['proba'][n]):.4f})")
    ax.plot([0,1],[0,1],"--",c="gray"); ax.set(title=f"ROC curve - {NAMES[d]}", xlabel="False positive rate", ylabel="True positive rate"); ax.legend(loc="lower right")
    plt.tight_layout(); save(fig, f"{d[-1]}_5_roc_curve")

# ---------------- extra graphs: confusion matrices, PR, calibration, score distributions ----------------
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(2, 3, figsize=(15, 9)); fig.suptitle("Confusion matrices - every model on both datasets (test, tuned threshold; count and row %)", fontsize=14, fontweight="bold")
for i, d in enumerate((d1, d2)):
    for j, n in enumerate(MODELS):
        a = ax[i, j]; a.grid(False); cm = confusion_matrix(R[d]["te_y"], (R[d]["proba"][n] >= R[d]["thr"][n]).astype(int)); pc = cm / cm.sum(1, keepdims=True)
        a.imshow(pc, cmap="Blues", vmin=0, vmax=1)
        for u in range(2):
            for v in range(2): a.text(v, u, f"{cm[u,v]:,}\n{pc[u,v]:.1%}", ha="center", va="center", fontsize=12, fontweight="bold", color="w" if pc[u,v] > .5 else "k")
        a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xticklabels(["No purchase","Purchase"]); a.set_yticklabels(["No purchase","Purchase"])
        a.set_title(f"{d} - {n}" + ("  [FINAL]" if n == R[d]["final"] else ""), fontsize=10); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); save(fig, "C6_confusion_matrices_all_models")
for d in (d1, d2):
    fig, ax = plt.subplots(figsize=(7, 6))
    for n in MODELS:
        p, rc, _ = precision_recall_curve(R[d]["te_y"], R[d]["proba"][n]); ax.plot(rc, p, lw=2.6, color=MC[n], label=f"{n} (AP {average_precision_score(R[d]['te_y'], R[d]['proba'][n]):.4f})")
    ax.axhline(R[d]["te_y"].mean(), ls="--", c="gray", label="Random baseline"); ax.set(title=f"Precision-Recall curve - {NAMES[d]}", xlabel="Recall", ylabel="Precision"); ax.legend(loc="lower left")
    plt.tight_layout(); save(fig, f"{d[-1]}_6_pr_curve")
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5)); fig.suptitle("Calibration (reliability) curves - do predicted probabilities match reality? (test)", fontsize=14, fontweight="bold")
for a, d in zip(ax, (d1, d2)):
    for n in MODELS:
        fr, mp = calibration_curve(R[d]["te_y"], R[d]["proba"][n], n_bins=10, strategy="quantile"); a.plot(mp, fr, marker="o", lw=2, color=MC[n], label=n)
    a.plot([0,1],[0,1],"--",c="gray",label="Perfect"); a.set(title=NAMES[d], xlabel="Mean predicted probability", ylabel="Observed purchase rate"); a.legend()
plt.tight_layout(); save(fig, "C7_calibration_curves")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8)); fig.suptitle("Predicted purchase probability by true class - final model (test)", fontsize=14, fontweight="bold")
for a, d in zip(ax, (d1, d2)):
    fm = R[d]["final"]; p = R[d]["proba"][fm]; yt = R[d]["te_y"].values; bins = np.linspace(0, 1, 41)
    a.hist(p[yt == 0], bins=bins, alpha=.75, color="#B0B7C3", label="No purchase", log=True); a.hist(p[yt == 1], bins=bins, alpha=.8, color=DC[d], label="Purchase", log=True)
    a.axvline(R[d]["thr"][fm], ls="--", c="k", label=f"threshold {R[d]['thr'][fm]:.2f}"); a.set(title=f"{d} - {fm}", xlabel="predicted probability", ylabel="count (log)"); a.legend()
plt.tight_layout(); save(fig, "C8_probability_distribution")
fig, ax = plt.subplots(1, 2, figsize=(15, 5)); fig.suptitle("Validation vs Test F1 (tuned threshold) - generalisation check", fontsize=14, fontweight="bold")
V = M[(M.Split == "Validation") & (M.Threshold == "tuned")].set_index(["Dataset", "Model"])
for a, d in zip(ax, (d1, d2)):
    xx = np.arange(3)
    for k, (lab, df_, c) in enumerate((("Validation", V, "#9AA0A6"), ("Test", T, DC[d]))):
        v = [df_.loc[(d, n), "F1"] for n in MODELS]; b = a.bar(xx+(k-.5)*.36, v, .36, label=lab, color=c)
        for bb, vv in zip(b, v): a.text(bb.get_x()+bb.get_width()/2, vv+.005, f"{vv:.3f}", ha="center", fontsize=9)
    a.set_xticks(xx); a.set_xticklabels(MODELS); a.set(title=NAMES[d], ylim=(0, 1.08), ylabel="F1"); a.legend(loc="lower right")
plt.tight_layout(); save(fig, "C9_validation_vs_test_f1")

with PdfPages(OUT + "/all_graphs_report.pdf") as pdf:
    for f in figs: pdf.savefig(f, bbox_inches="tight")
json.dump({"overall_best": overall_best, "final": {d: R[d]["final"] for d in R}, "thr": {d: R[d]["thr"] for d in R}}, open(OUT + "/summary.json", "w"), indent=1)
print("DONE overall best:", overall_best)
