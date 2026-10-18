"""
dataset1_train.py
Dataset 1 ONLY: clickstream purchase-propensity data (target = 'ordered').

Run on its own:  python dataset1_train.py
Does: load -> clean -> feature engineer -> train all 5 models -> tune
threshold -> evaluate -> save dataset-1-only graphs + metrics + a results
pickle for combined_comparison.py. Does NOT touch dataset 2 at all.
"""
import os
import pandas as pd
import common

DS_KEY = "ds1"
DS_NAME = "Dataset 1 (Clickstream, 'ordered')"
COLOR = "#2A9D8F"

os.makedirs(common.G, exist_ok=True)


def load_and_clean():
    df = pd.read_csv("data/dataset1_clickstream_training_sample.csv")
    n0 = len(df)
    df = df.drop_duplicates(subset="UserID")
    print(f"Dataset 1: removed {n0 - len(df)} duplicate rows (by UserID)")

    y = df["ordered"]
    X = df.drop(columns=["UserID", "ordered"])

    # engineered behavioural features
    X["total_clicks"] = X[[
        "basket_icon_click", "basket_add_list", "basket_add_detail", "sort_by", "image_picker",
        "account_page_click", "promo_banner_click", "detail_wishlist_add", "list_size_dropdown",
        "closed_minibasket_click",
    ]].sum(axis=1)
    X["info_checks"] = X[[
        "checked_delivery_detail", "checked_returns_detail", "saw_sizecharts", "saw_delivery",
    ]].sum(axis=1)
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
    print("\nDataset 1 DONE. Final model:", r["final"])