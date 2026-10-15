import sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

warnings.filterwarnings("ignore")
PATH = sys.argv[1] if len(sys.argv) > 1 else "training_sample.csv"
SEED, THRESHOLD, TARGET = 42, 0.5, "ordered"

# ───────────────────────── KDD 1: SELECTION ─────────────────────────
print("=" * 60, "\nSTEP 1: DATA SELECTION\n" + "=" * 60)
df = pd.read_csv(PATH)
print(f"Shape: {df.shape}")
print(f"Buyers: {df[TARGET].mean():.2%}  (imbalanced)")
print(f"Duplicate UserIDs: {df['UserID'].duplicated().sum():,}")
print("Note: testing_sample.csv contains no buyers, so all evaluation "
      "is done on splits of this file.")

# ───────────────────────── KDD 2: PREPROCESSING ─────────────────────
print("\n" + "=" * 60, "\nSTEP 2: PREPROCESSING / CLEANING\n" + "=" * 60)
print("Missing values:", int(df.isna().sum().sum()))
before = len(df)
df = df.drop_duplicates(subset="UserID")
print(f"Rows dropped (duplicate users): {before - len(df)}")
features = [c for c in df.columns if c not in ("UserID", TARGET)]
print(f"Binary/count features: {len(features)}")

# ───────────────────────── KDD 3: TRANSFORMATION ────────────────────
print("\n" + "=" * 60, "\nSTEP 3: TRANSFORMATION\n" + "=" * 60)
# Engineered features: overall activity and checkout-funnel depth
df["total_clicks"] = df[["basket_icon_click", "basket_add_list", "basket_add_detail",
                         "sort_by", "image_picker", "account_page_click",
                         "promo_banner_click", "detail_wishlist_add",
                         "list_size_dropdown", "closed_minibasket_click"]].sum(axis=1)
df["info_checks"] = df[["checked_delivery_detail", "checked_returns_detail",
                        "saw_sizecharts", "saw_delivery"]].sum(axis=1)
features += ["total_clicks", "info_checks"]
print("Added features: total_clicks, info_checks")

# 60 / 20 / 20 stratified split: train / validation / test
train_val, test = train_test_split(df, test_size=0.20, stratify=df[TARGET], random_state=SEED)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[TARGET], random_state=SEED)
print(f"Train {len(train):,} | Validation {len(val):,} | Test {len(test):,}")
X_tr, y_tr = train[features], train[TARGET]
X_va, y_va = val[features], val[TARGET]
X_te, y_te = test[features], test[TARGET]

# Class imbalance handled with class weights (no synthetic data needed,
# and validation/test stay untouched and realistic)
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

# ───────────────────────── KDD 4: DATA MINING ───────────────────────
print("\n" + "=" * 60, "\nSTEP 4: DATA MINING (MODEL TRAINING)\n" + "=" * 60)
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=14, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=pos_weight,
        n_jobs=-1, random_state=SEED, eval_metric="aucpr"),
}
fitted = {}
for name, m in models.items():
    t0 = time.time()
    m.fit(X_tr, y_tr)
    fitted[name] = m
    print(f"{name:<22} trained in {time.time() - t0:5.1f}s")

# ───────────────────────── KDD 5: EVALUATION ────────────────────────
def metrics(y, proba, thr=THRESHOLD):
    pred = (proba >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba),
            "PR-AUC": average_precision_score(y, proba)}

print("\n" + "=" * 60, f"\nSTEP 5: EVALUATION (validation set, threshold={THRESHOLD})\n" + "=" * 60)
val_proba = {n: m.predict_proba(X_va)[:, 1] for n, m in fitted.items()}
val_table = pd.DataFrame({n: metrics(y_va, p) for n, p in val_proba.items()}).T
print((val_table * 100).round(2).to_string())

# Final model = highest validation F1 (balances precision and recall)
best = val_table["F1"].idxmax()
print(f"\n>>> FINAL MODEL CHOSEN: {best} (highest validation F1 = {val_table.loc[best,'F1']:.2%})")

# One-time test evaluation of the chosen model
print("\n" + "=" * 60, f"\nFINAL TEST RESULTS: {best} (threshold={THRESHOLD})\n" + "=" * 60)
te_proba = fitted[best].predict_proba(X_te)[:, 1]
final = metrics(y_te, te_proba)
for k, v in final.items():
    print(f"{k:<10}: {v:.2%}")
cm = confusion_matrix(y_te, (te_proba >= THRESHOLD).astype(int))
print("\nConfusion matrix [[TN FP],[FN TP]]:\n", cm)

# Interpretation: feature importance
m = fitted[best]
if best == "Logistic Regression":
    imp = pd.Series(np.abs(m[-1].coef_[0]), features)
else:
    imp = pd.Series(m.feature_importances_, features)
imp = (imp / imp.sum()).sort_values(ascending=False)
print("\nTop features:\n", (imp.head(8) * 100).round(2).to_string())
print("\nWARNING: if two features dominate (e.g. checkout/delivery pages), "
      "the model mainly detects users already near purchase.")

# ───────────────── Plots (interpretation) ─────────────────
fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
for n, p in val_proba.items():
    f, t, _ = roc_curve(y_va, p); ax[0].plot(f, t, label=f"{n} ({roc_auc_score(y_va,p):.3f})")
    pr, rc, _ = precision_recall_curve(y_va, p); ax[1].plot(rc, pr, label=n)
ax[0].plot([0, 1], [0, 1], "--", c="gray"); ax[0].set(title="ROC (validation)", xlabel="FPR", ylabel="TPR"); ax[0].legend()
ax[1].set(title="Precision-Recall (validation)", xlabel="Recall", ylabel="Precision"); ax[1].legend()
(val_table[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]] * 100).plot.bar(ax=ax[2], rot=15)
ax[2].set(title="Validation metrics (%)", ylim=(0, 105))
imp.head(10).iloc[::-1].plot.barh(ax=ax[3], title=f"{best}: top features")
plt.tight_layout(); plt.savefig("kdd_results.png", dpi=150)

val_table.round(4).to_csv("validation_comparison.csv")
pd.Series(final).round(4).to_csv("final_test_metrics.csv")
print("\nSaved: kdd_results.png, validation_comparison.csv, final_test_metrics.csv")
