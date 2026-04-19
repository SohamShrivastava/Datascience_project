import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from evaluation.metrics import rmse, mae
from models.baseline import BaselineModel
from models.matrix_factorization import MatrixFactorization
from models.svdpp import SVDPP


def evaluate_models(df, pre):
    # split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    results = []

    # ---------- BASELINE ----------
    base = BaselineModel()
    base.fit(train_df)

    y_true = []
    y_pred = []

    for row in test_df.itertuples():
        pred = base.predict_bias(row.user, row.item)
        y_true.append(row.rating)
        y_pred.append(pred)

    results.append({
        "model": "Baseline",
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    })

    # ---------- MATRIX FACTORIZATION ----------
    n_users, n_items = pre.get_num_users_items(df)

    mf = MatrixFactorization(n_users, n_items, epochs=5)
    mf.fit(train_df)

    y_true = []
    y_pred = []

    for row in test_df.itertuples():
        pred = mf.predict(row.user, row.item)
        y_true.append(row.rating)
        y_pred.append(pred)

    results.append({
        "model": "MatrixFactorization",
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    })

    # ---------- SVD++ ----------
    print("Starting SVD++...", flush=True)
    svdpp = SVDPP(n_users, n_items, epochs=2)
    svdpp.fit(train_df)

    y_true = []
    y_pred = []

    for row in test_df.itertuples():
        pred = svdpp.predict(row.user, row.item)
        y_true.append(row.rating)
        y_pred.append(pred)

    results.append({
        "model": "SVD++",
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    })
    print("Finished SVD++", flush=True)

    # save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("./outputs/results.csv", index=False)

    return results_df