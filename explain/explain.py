import numpy as np


class Explainer:
    def __init__(self, df, movies_df, mf_model):
        self.df = df
        self.movies = movies_df
        self.mf = mf_model

        # movieId → title
        self.movie_titles = dict(zip(movies_df.movieId, movies_df.title))

        # movieId → genres
        self.movie_genres = {
            row.movieId: set(row.genres.split("|"))
            for row in movies_df.itertuples()
        }

    def get_user_history(self, user, top_n=5):
        user_data = self.df[self.df['user'] == user]
        liked = user_data.sort_values(by='rating', ascending=False).head(top_n)

        return liked['movieId'].tolist()

    def genre_overlap(self, movie1, movie2):
        g1 = self.movie_genres.get(movie1, set())
        g2 = self.movie_genres.get(movie2, set())

        return len(g1 & g2)

    def explain(self, user, item, pre):
        """
        item = encoded item
        """
        # convert to original movieId
        movieId = pre.movie_encoder.inverse_transform([item])[0]

        # get user history
        history = self.get_user_history(user)

        # find similar movies from history (genre-based)
        similarities = []
        for m in history:
            overlap = self.genre_overlap(movieId, m)
            similarities.append((m, overlap))

        # sort by overlap
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        top_similar = [self.movie_titles[m] for m, _ in similarities[:3]]

        # genre match
        genres = self.movie_genres.get(movieId, set())

        # collaborative signal
        score = self.mf.predict(user, item)

        explanation = f"""
Recommended: {self.movie_titles.get(movieId, "Unknown")}

Because:
- You liked: {", ".join(top_similar)}
- Genre match: {", ".join(genres)}
- Predicted rating (MF): {round(score, 2)}
"""
        return explanation