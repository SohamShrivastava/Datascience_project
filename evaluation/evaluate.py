import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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


def evaluate_models(df, pre, movies_df_param):
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


    # ---------- KNN ----------
    from models.knn import KNNModel
    knn = KNNModel(k=20, min_common=3)
    knn.fit(train_df)

    y_true, y_pred = [], []
    for row in test_df.itertuples():
        y_true.append(row.rating)
        y_pred.append(knn.predict(row.user, row.item))

    results.append({
        "model": "KNN",
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    })

    # ---------- HYBRID ----------
    from models.hybrid import HybridRecommender
    user_history = {}
    for row in train_df.itertuples():
        user_history.setdefault(row.user, []).append(row.movieId)

    hybrid = HybridRecommender(mf, movies_df_param, alpha=0.9, preprocessor=pre)

    y_true, y_pred = [], []
    for row in test_df.itertuples():
        history = user_history.get(row.user, [])
        movieId = pre.movie_encoder.inverse_transform([row.item])[0]
        y_true.append(row.rating)
        y_pred.append(hybrid.predict(row.user, movieId, history))

    results.append({
        "model": "Hybrid",
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    })




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


def _fast_diversity_at_k(recommended, genre_map, k=10):
    recommended_k = list(recommended[:k])

    if len(recommended_k) < 2:
        return 0.0

    pair_distances = []
    n = len(recommended_k)
    for i in range(n):
        genres_a = genre_map.get(recommended_k[i], set())
        for j in range(i + 1, n):
            genres_b = genre_map.get(recommended_k[j], set())
            union = genres_a | genres_b
            if not union:
                pair_distances.append(0.0)
                continue

            similarity = len(genres_a & genres_b) / len(union)
            pair_distances.append(1.0 - similarity)

    return float(np.mean(pair_distances)) if pair_distances else 0.0


def evaluate_ranking_model(
    model,
    train_df,
    test_df,
    pre,
    movies_df,
    k=10,
    rating_threshold=4.0,
    candidate_sample_size=100,
    max_users=None,
    random_state=42,
    show_progress=True,
    candidate_provider=None,
):
    """
    Evaluate top-k recommendation quality using held-out positive interactions.
    """
    rng = random.Random(random_state)
    popularity_map = train_df.groupby('movieId').size().to_dict()
    # Build once for all users in this model evaluation.
    genre_map = {
        row.movieId: set(str(row.genres).split("|"))
        for row in movies_df.itertuples()
    }
    item_to_movie = np.asarray(pre.movie_encoder.classes_)
    users = sorted(test_df['user'].unique())
    if max_users is not None:
        users = users[:max_users]

    precision_scores = []
    recall_scores = []
    diversity_scores = []
    novelty_scores = []
    serendipity_scores = []

    n_items = pre.get_num_users_items(train_df)[1]

    user_iterator = users
    if show_progress:
        user_iterator = tqdm(
            users,
            total=len(users),
            desc=f"Evaluating {model.__class__.__name__}",
            unit="user",
        )

    for user in user_iterator:
        train_items = set(train_df.loc[train_df['user'] == user, 'item'].tolist())
        user_test = test_df[test_df['user'] == user]
        relevant_items = user_test.loc[user_test['rating'] >= rating_threshold, 'item'].tolist()
        
        if candidate_provider is not None:
            candidate_items = candidate_provider(user, top_n=candidate_sample_size)
        else:
            all_items = set(range(n_items)) - train_items
            sample_size = min(candidate_sample_size, len(all_items))
            candidate_items = rng.sample(list(all_items), sample_size) if sample_size > 0 else []


        # # candidate_items = [item for item in range(n_items) if item not in train_items]
        # all_items = set(range(n_items)) - train_items
        # sample_size = min(candidate_sample_size, len(all_items))
        # candidate_items = rng.sample(list(all_items), sample_size) if sample_size > 0 else []
        # Always include relevant items
        candidate_items = list(set(candidate_items) | set(relevant_items))
        scored_items = []

        for item in candidate_items:
            try:
                score = _score_item(model, user, item)
            except Exception:
                continue

            scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in scored_items[:k]]

        recommended_movie_ids = item_to_movie[recommended_items].tolist() if recommended_items else []
        relevant_movie_ids = item_to_movie[relevant_items].tolist() if relevant_items else []

        precision_scores.append(precision_at_k(recommended_movie_ids, relevant_movie_ids, k=k))
        recall_scores.append(recall_at_k(recommended_movie_ids, relevant_movie_ids, k=k))
        diversity_scores.append(_fast_diversity_at_k(recommended_movie_ids, genre_map, k=k))
        novelty_scores.append(novelty_at_k(recommended_movie_ids, popularity_map, k=k))
        serendipity_scores.append(float((diversity_scores[-1] + novelty_scores[-1]) / 2.0))

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