import argparse
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from evaluation.metrics import rmse, mae
from models.hybrid import HybridRecommender
from models.matrix_factorization import MatrixFactorization
from src.data_loader import load_and_merge
from src.preprocessing import Preprocessor


def build_user_history(train_df):
    user_history = {}

    for row in train_df.itertuples():
        user_history.setdefault(row.user, []).append(row.movieId)

    return user_history


def evaluate_mf(train_df, valid_df, pre, params):
    n_users, n_items = pre.get_num_users_items(train_df)
    model = MatrixFactorization(n_users, n_items, **params)
    model.fit(train_df)

    y_true = []
    y_pred = []

    for row in valid_df.itertuples():
        y_true.append(row.rating)
        y_pred.append(model.predict(row.user, row.item))

    return {
        "params": params,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "model": model,
    }


def evaluate_hybrid(valid_df, model, movies, user_history, alpha, preprocessor):
    hybrid = HybridRecommender(model, movies, alpha=alpha, preprocessor=preprocessor)

    y_true = []
    y_pred = []

    for row in valid_df.itertuples():
        history = user_history.get(row.user, [])
        movie_id = row.movieId
        y_true.append(row.rating)
        y_pred.append(hybrid.predict(row.user, movie_id, history))

    return {
        "alpha": alpha,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Tune MF and hybrid hyperparameters.")
    parser.add_argument("--movies", default="./data/movies.csv")
    parser.add_argument("--ratings", default="./data/ratings.csv")
    parser.add_argument("--output", default="./outputs/tuning_results.csv")
    args = parser.parse_args()

    df = load_and_merge(args.movies, args.ratings)

    pre = Preprocessor()
    pre.fit_ids(df)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = pre.transform_ids(train_df)
    valid_df = pre.transform_ids(valid_df)

    movies = pd.read_csv(args.movies)
    user_history = build_user_history(train_df)

    mf_grid = {
        "n_factors": [20, 40],
        "epochs": [5, 10],
        "lr": [0.005, 0.01],
        "reg": [0.02, 0.05],
        "decay_rate": [0.0, 0.01],
    }

    mf_results = []
    mf_param_names = list(mf_grid.keys())
    mf_combinations = list(itertools.product(*mf_grid.values()))

    for trial_idx, values in enumerate(tqdm(mf_combinations, desc="MF tuning", unit="trial"), start=1):
        params = dict(zip(mf_param_names, values))
        print(f"\n[MF Trial {trial_idx}/{len(mf_combinations)}] params={params}")
        result = evaluate_mf(train_df, valid_df, pre, params)
        mf_results.append(result)

    mf_results_df = pd.DataFrame([
        {**{k: v for k, v in item["params"].items()}, "RMSE": item["RMSE"], "MAE": item["MAE"]}
        for item in mf_results
    ])
    mf_results_df = mf_results_df.dropna(subset=["RMSE", "MAE"])

    if mf_results_df.empty:
        raise ValueError("All MF trials produced invalid metrics. Check data quality and training settings.")

    best_mf_row = mf_results_df.sort_values(["RMSE", "MAE"]).iloc[0]

    best_params = {
        "n_factors": int(best_mf_row["n_factors"]),
        "epochs": int(best_mf_row["epochs"]),
        "lr": float(best_mf_row["lr"]),
        "reg": float(best_mf_row["reg"]),
        "decay_rate": float(best_mf_row["decay_rate"]),
    }

    n_users, n_items = pre.get_num_users_items(train_df)
    best_mf = MatrixFactorization(n_users, n_items, **best_params)
    best_mf.fit(train_df)

    hybrid_results = []
    for alpha in tqdm([0.1, 0.3, 0.5, 0.7, 0.9], desc="Hybrid alpha", unit="alpha"):
        hybrid_results.append(evaluate_hybrid(valid_df, best_mf, movies, user_history, alpha, pre))

    hybrid_results_df = pd.DataFrame(hybrid_results)

    output_df = pd.concat(
        [
            mf_results_df.assign(search_space="matrix_factorization"),
            hybrid_results_df.assign(search_space="hybrid_alpha"),
        ],
        ignore_index=True,
        sort=False,
    )
    output_df.to_csv(args.output, index=False)

    print("Best MF params:")
    print(best_mf_row.to_dict())
    print("\nHybrid alpha results:")
    print(hybrid_results_df.sort_values(["RMSE", "MAE"]))


if __name__ == "__main__":
    main()