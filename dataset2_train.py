"""
dataset2_train.py
Dataset 2 ONLY: UCI Online Shoppers Intention data (target = 'Revenue').

Run on its own:  python dataset2_train.py
Does: load -> clean -> feature engineer -> train all 5 models -> tune
threshold -> evaluate -> save dataset-2-only graphs + metrics + a results
pickle for combined_comparison.py. Does NOT touch dataset 1 at all.
"""
import os
import numpy as np, pandas as pd
import common

DS_KEY = "ds2"
DS_NAME = "Dataset 2 (UCI Online Shoppers, 'Revenue')"
COLOR = "#E76F51"

os.makedirs(common.G, exist_ok=True)


def load_and_clean():
    df = pd.read_csv("data/dataset2_online_shoppers_intention.csv")
    n0 = len(df)
    df = df.drop_duplicates()
    print(f"Dataset 2: removed {n0 - len(df)} duplicate rows")

    y = df["Revenue"].astype(str).str.upper().eq("TRUE").astype(int)
    X = df.drop(columns="Revenue").copy()
    X["Weekend"] = X["Weekend"].astype(str).str.upper().eq("TRUE").astype(int)

    # engineered behavioural features
    X["TotalPages"] = X.Administrative + X.Informational + X.ProductRelated
    X["TotalDuration"] = X.Administrative_Duration + X.Informational_Duration + X.ProductRelated_Duration
    X["AvgTimePerPage"] = (X.TotalDuration / X.TotalPages.replace(0, np.nan)).fillna(0)
    X["ExitBounceDiff"] = X.ExitRates - X.BounceRates
    X["PageValuePerProductPage"] = (X.PageValues / X.ProductRelated.replace(0, np.nan)).fillna(0)

    X = pd.get_dummies(X, columns=["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"], dtype=int)
    return X, y


if __name__ == "__main__":
    X, y = load_and_clean()
    r = common.run_training(DS_KEY, DS_NAME, X, y)

    common.plot_class_balance(DS_KEY, DS_NAME, COLOR, y)
    common.plot_data_split(DS_KEY, DS_NAME, COLOR, r["sizes"])
    common.plot_top_correlations(DS_KEY, DS_NAME, COLOR, X, y)
    common.plot_metrics_bars(DS_KEY, DS_NAME, r)
    common.plot_roc_curves(DS_KEY, DS_NAME, r)
    common.plot_precision_recall(DS_KEY, DS_NAME, r)
    common.plot_threshold_tuning(DS_KEY, DS_NAME, COLOR, r)
    common.plot_confusion_matrix(DS_KEY, DS_NAME, r)
    common.plot_feature_importance(DS_KEY, DS_NAME, COLOR, r)

    common.save_dataset_outputs(DS_KEY, r)
    print("\nDataset 2 DONE. Final model:", r["final"])