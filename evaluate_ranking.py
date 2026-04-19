import pandas as pd

from evaluation.evaluate import evaluate_ranking_model
from models.baseline import BaselineModel
from models.matrix_factorization import MatrixFactorization
from models.svdpp import SVDPP
from src.data_loader import load_and_merge
from src.preprocessing import Preprocessor


def main():
    df = load_and_merge("./data/movies.csv", "./data/ratings.csv")

    pre = Preprocessor()
    df = pre.encode_ids(df)

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    movies_df = pd.read_csv("./data/movies.csv")
    n_users, n_items = pre.get_num_users_items(df)

    baseline = BaselineModel()
    baseline.fit(train_df)

    mf = MatrixFactorization(n_users, n_items, epochs=5, decay_rate=0.01)
    mf.fit(train_df)

    svdpp = SVDPP(n_users, n_items, epochs=2)
    svdpp.fit(train_df)

    ranking_results = pd.concat(
        [
            evaluate_ranking_model(baseline, train_df, test_df, pre, movies_df, k=10),
            evaluate_ranking_model(mf, train_df, test_df, pre, movies_df, k=10),
            evaluate_ranking_model(svdpp, train_df, test_df, pre, movies_df, k=10),
        ],
        ignore_index=True,
    )

    ranking_results.to_csv("./outputs/ranking_results.csv", index=False)
    print(ranking_results)


if __name__ == "__main__":
    main()