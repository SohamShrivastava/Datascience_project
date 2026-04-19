import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from evaluation.metrics import (
    rmse,
    mae,
    precision_at_k,
    recall_at_k,
    diversity_at_k,
    novelty_at_k,
    serendipity_proxy_at_k,
)
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


def _score_item(model, user, item):
    if hasattr(model, "predict"):
        return model.predict(user, item)

    if hasattr(model, "predict_bias"):
        return model.predict_bias(user, item)

    raise AttributeError("Model must define predict() or predict_bias().")


def evaluate_ranking_model(model, train_df, test_df, pre, movies_df, k=10, rating_threshold=4.0):
    """
    Evaluate top-k recommendation quality using held-out positive interactions.
    """
    popularity_map = train_df.groupby('movieId').size().to_dict()
    users = sorted(test_df['user'].unique())

    precision_scores = []
    recall_scores = []
    diversity_scores = []
    novelty_scores = []
    serendipity_scores = []

    n_items = pre.get_num_users_items(train_df)[1]

    for user in users:
        train_items = set(train_df.loc[train_df['user'] == user, 'item'].tolist())
        user_test = test_df[test_df['user'] == user]
        relevant_items = user_test.loc[user_test['rating'] >= rating_threshold, 'item'].tolist()

        candidate_items = [item for item in range(n_items) if item not in train_items]
        scored_items = []

        for item in candidate_items:
            try:
                score = _score_item(model, user, item)
            except Exception:
                continue

            scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in scored_items[:k]]

        recommended_movie_ids = pre.movie_encoder.inverse_transform(recommended_items) if recommended_items else []
        relevant_movie_ids = pre.movie_encoder.inverse_transform(relevant_items) if relevant_items else []

        precision_scores.append(precision_at_k(recommended_movie_ids, relevant_movie_ids, k=k))
        recall_scores.append(recall_at_k(recommended_movie_ids, relevant_movie_ids, k=k))
        diversity_scores.append(diversity_at_k(recommended_movie_ids, movies_df, k=k))
        novelty_scores.append(novelty_at_k(recommended_movie_ids, popularity_map, k=k))
        serendipity_scores.append(serendipity_proxy_at_k(recommended_movie_ids, movies_df, popularity_map, k=k))

    return pd.DataFrame([
        {
            "model": model.__class__.__name__,
            f"Precision@{k}": float(np.mean(precision_scores)) if precision_scores else 0.0,
            f"Recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
            f"Diversity@{k}": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
            f"Novelty@{k}": float(np.mean(novelty_scores)) if novelty_scores else 0.0,
            f"SerendipityProxy@{k}": float(np.mean(serendipity_scores)) if serendipity_scores else 0.0,
        }
    ])