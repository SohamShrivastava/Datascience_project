import pandas as pd


class ColdStartRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df

        # popularity = number of ratings
        self.popularity = ratings_df.groupby('movieId').size().reset_index(name='count')

        # merge with movies
        self.popular_movies = self.popularity.merge(self.movies, on='movieId')

    def recommend(self, genres, top_n=10):
        """
        genres = list of genres selected by user
        """

        # filter movies containing any selected genre
        filtered = self.popular_movies[
            self.popular_movies['genres'].apply(
                lambda x: any(g in x for g in genres)
            )
        ]

        # sort by popularity
        filtered = filtered.sort_values(by='count', ascending=False)

        return filtered[['title', 'genres']].head(top_n)
    
    def recommend_from_movies(self, movie_titles, top_n=10):


        # get selected movieIds
        selected = self.movies[self.movies['title'].isin(movie_titles)]

        if len(selected) == 0:
            return pd.DataFrame()

        # get their genres
        selected_genres = set()
        for g in selected['genres']:
            selected_genres.update(g.split("|"))

        # filter similar movies
        filtered = self.popular_movies[
            self.popular_movies['genres'].apply(
                lambda x: any(g in x for g in selected_genres)
            )
        ]

        # remove already selected movies
        filtered = filtered[~filtered['title'].isin(movie_titles)]

        # sort by popularity
        filtered = filtered.sort_values(by='count', ascending=False)

        return filtered[['title', 'genres']].head(top_n)