import argparse
import time

import pandas as pd

from evaluation.evaluate import evaluate_ranking_model
from models.baseline import BaselineModel
from models.matrix_factorization import MatrixFactorization
from models.svdpp import SVDPP
from src.data_loader import load_and_merge
from src.preprocessing import Preprocessor
from models.knn import KNNModel
from models.hybrid import HybridRecommender

def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking metrics for recommendation models.")
    parser.add_argument("--movies", default="./data/movies.csv")
    parser.add_argument("--ratings", default="./data/ratings.csv")
    parser.add_argument("--output", default="./outputs/ranking_results.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--train-frac", type=float, default=1.0, help="Fraction of train split used for fitting models.")
    parser.add_argument("--candidate-sample-size", type=int, default=100)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--mf-epochs", type=int, default=5)
    parser.add_argument("--svdpp-epochs", type=int, default=2)
    parser.add_argument(
        "--svdpp-max-implicit-items",
        type=int,
        default=None,
        help="Cap implicit history size per user in SVD++ to speed up training.",
    )
    parser.add_argument("--skip-svdpp", action="store_true", help="Skip SVD++ model (useful for fast validation).")
    parser.add_argument("--quick", action="store_true", help="Fast sanity-check mode.")
    args = parser.parse_args()

    if args.quick:
        # Override heavy settings for quick validation.
        args.max_users = 10000 if args.max_users is None else args.max_users
        args.candidate_sample_size = min(args.candidate_sample_size, 30)
        args.mf_epochs = min(args.mf_epochs, 1)
        args.svdpp_epochs = min(args.svdpp_epochs, 1)
        args.train_frac = min(args.train_frac, 0.25)
        args.svdpp_max_implicit_items = 50 if args.svdpp_max_implicit_items is None else args.svdpp_max_implicit_items
        args.skip_svdpp = True

    df = load_and_merge(args.movies, args.ratings)

    pre = Preprocessor()
    df = pre.encode_ids(df)

    train_df = df.sample(frac=1.0 - args.test_frac, random_state=42)
    test_df = df.drop(train_df.index)
    if args.train_frac < 1.0:
        train_df = train_df.sample(frac=args.train_frac, random_state=42)

    print(
        f"Train rows: {len(train_df)}, Test rows: {len(test_df)}, "
        f"Max users eval: {args.max_users if args.max_users is not None else 'all'}"
    )

    movies_df = pd.read_csv(args.movies)
    n_users, n_items = pre.get_num_users_items(df)

    ranking_frames = []

    baseline = BaselineModel()
    t0 = time.perf_counter()
    print("Starting Baseline fit...")
    baseline.fit(train_df)
    print(f"Baseline fit done in {time.perf_counter() - t0:.2f}s")
    ranking_frames.append(
        evaluate_ranking_model(
            baseline,
            train_df,
            test_df,
            pre,
            movies_df,
            k=args.k,
            candidate_sample_size=args.candidate_sample_size,
            max_users=args.max_users,
            random_state=42,
        )
    )

    mf = MatrixFactorization(n_users, n_items, epochs=args.mf_epochs, decay_rate=0.01)
    t0 = time.perf_counter()
    print("Starting MF fit...")
    mf.fit(train_df)
    print(f"MF fit done in {time.perf_counter() - t0:.2f}s")
    ranking_frames.append(
        evaluate_ranking_model(
            mf,
            train_df,
            test_df,
            pre,
            movies_df,
            k=args.k,
            candidate_sample_size=args.candidate_sample_size,
            max_users=args.max_users,
            random_state=42,
        )
    )
    # Build user history for hybrid
    user_history = {}
    for row in train_df.itertuples():
        user_history.setdefault(row.user, []).append(row.movieId)

    hybrid = HybridRecommender(mf, movies_df, alpha=0.9, preprocessor=pre)

    # Wrap predict to match evaluate_ranking_model signature
    class HybridWrapper:
        def __init__(self, hybrid_model, user_history):
            self.hybrid = hybrid_model
            self.user_history = user_history
            self.__class__.__name__ = "HybridModel"

        def predict(self, user, item):
            history = self.user_history.get(user, [])
            return self.hybrid.predict(user, item, history)

    hybrid_wrapper = HybridWrapper(hybrid, user_history)

    t0 = time.perf_counter()
    print("Starting Hybrid evaluation...")
    ranking_frames.append(
        evaluate_ranking_model(
            hybrid_wrapper,
            train_df,
            test_df,
            pre,
            movies_df,
            k=args.k,
            candidate_sample_size=args.candidate_sample_size,
            max_users=args.max_users,
            random_state=42,
        )
    )
    print(f"Hybrid evaluation done in {time.perf_counter() - t0:.2f}s")

    if not args.skip_svdpp:
        svdpp = SVDPP(
            n_users,
            n_items,
            epochs=args.svdpp_epochs,
            max_implicit_items=args.svdpp_max_implicit_items,
            show_progress=True,
        )
        t0 = time.perf_counter()
        print("Starting SVD++ fit... (this is the slowest stage)")
        svdpp.fit(train_df)
        print(f"SVD++ fit done in {time.perf_counter() - t0:.2f}s")
        ranking_frames.append(
            evaluate_ranking_model(
                svdpp,
                train_df,
                test_df,
                pre,
                movies_df,
                k=args.k,
                candidate_sample_size=args.candidate_sample_size,
                max_users=args.max_users,
                random_state=42,
            )
        )
    
    knn = KNNModel(k=20, min_common=3)
    t0 = time.perf_counter()
    print("Starting KNN fit...")
    knn.fit(train_df)
    print(f"KNN fit done in {time.perf_counter() - t0:.2f}s")
    ranking_frames.append(
        evaluate_ranking_model(
            knn,
            train_df,
            test_df,
            pre,
            movies_df,
            k=args.k,
            candidate_sample_size=args.candidate_sample_size,
            max_users=args.max_users,
            random_state=42,
            candidate_provider=knn.get_neighbor_items,  # ← KEY FIX
        )
    )
    
    ranking_results = pd.concat(ranking_frames, ignore_index=True)

    ranking_results.to_csv(args.output, index=False)
    print(ranking_results)


if __name__ == "__main__":
    main()