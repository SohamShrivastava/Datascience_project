import pandas as pd


def load_data(movies_path, ratings_path):
    """
    Load movies and ratings datasets
    """
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    return movies, ratings


def merge_data(movies, ratings):
    """
    Merge movies and ratings on movieId
    """
    df = pd.merge(ratings, movies, on='movieId')

    return df


def load_and_merge(movies_path, ratings_path):
    """
    Full pipeline
    """
    movies, ratings = load_data(movies_path, ratings_path)
    df = merge_data(movies, ratings)

    return df