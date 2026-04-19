import numpy as np


class HybridRecommender:
    def __init__(self, mf_model, movies_df, alpha=0.7):
        """
        alpha = weight for MF
        (1-alpha) = weight for genre similarity
        """
        self.mf = mf_model
        self.movies = movies_df
        self.alpha = alpha

        # create genre mapping
        self.movie_genres = {}
        for row in self.movies.itertuples():
            self.movie_genres[row.movieId] = set(row.genres.split("|"))

    def genre_similarity(self, movie1, movie2):
        """
        Jaccard similarity between genres
        """
        g1 = self.movie_genres.get(movie1, set())
        g2 = self.movie_genres.get(movie2, set())

        if len(g1 | g2) == 0:
            return 0

        return len(g1 & g2) / len(g1 | g2)

    def predict(self, user, item, user_history):
        """
        user_history = list of movieIds user liked
        """

        # MF prediction
        mf_score = self.mf.predict(user, item)

        # genre similarity score
        sims = []
        for m in user_history:
            sims.append(self.genre_similarity(item, m))

        genre_score = np.mean(sims) if sims else 0

        # final score
        genre_score_scaled = genre_score * 5  # bring to rating scale

        return self.alpha * mf_score + (1 - self.alpha) * genre_score_scaled