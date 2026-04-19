import numpy as np
from itertools import combinations


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = len([r for r in recommended_k if r in relevant_set])
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    if len(relevant_set) == 0:
        return 0

    hits = len([r for r in recommended_k if r in relevant_set])
    return hits / len(relevant_set)


def diversity_at_k(recommended, movies_df, k=10):
    """
    Average pairwise genre distance among the top-k recommendations.
    Returns values in [0, 1], where larger means more diverse.
    """
    recommended_k = list(recommended[:k])

    if len(recommended_k) < 2:
        return 0.0

    genre_map = {
        row.movieId: set(str(row.genres).split("|"))
        for row in movies_df.itertuples()
    }

    distances = []
    for movie_a, movie_b in combinations(recommended_k, 2):
        genres_a = genre_map.get(movie_a, set())
        genres_b = genre_map.get(movie_b, set())

        union = genres_a | genres_b
        if not union:
            distances.append(0.0)
            continue

        similarity = len(genres_a & genres_b) / len(union)
        distances.append(1 - similarity)

    return float(np.mean(distances)) if distances else 0.0


def novelty_at_k(recommended, popularity_map, k=10):
    """
    Simple novelty proxy based on inverse popularity.
    Higher values indicate less popular recommendations.
    """
    recommended_k = list(recommended[:k])

    if not recommended_k:
        return 0.0

    novelty_scores = []
    for item in recommended_k:
        popularity = popularity_map.get(item, 0)
        novelty_scores.append(1.0 / np.log2(2 + popularity))

    return float(np.mean(novelty_scores))


def serendipity_proxy_at_k(recommended, movies_df, popularity_map, k=10):
    """
    Practical serendipity proxy for offline evaluation.
    Combines diversity and novelty because true serendipity needs relevance labels.
    """
    diversity = diversity_at_k(recommended, movies_df, k=k)
    novelty = novelty_at_k(recommended, popularity_map, k=k)

    return float((diversity + novelty) / 2)