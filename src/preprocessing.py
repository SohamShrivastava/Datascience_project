import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

    def fit_ids(self, df):
        self.user_encoder.fit(df['userId'])
        self.movie_encoder.fit(df['movieId'])

        return self

    def transform_ids(self, df):
        df = df.copy()
        df['user'] = self.user_encoder.transform(df['userId'])
        df['item'] = self.movie_encoder.transform(df['movieId'])

        return df

    def encode_ids(self, df):
        df = df.copy()
        df['user'] = self.user_encoder.fit_transform(df['userId'])
        df['item'] = self.movie_encoder.fit_transform(df['movieId'])

        return df

    def create_user_item_matrix(self, df):
        user_item = df.pivot_table(
            index='user',
            columns='item',
            values='rating'
        ).fillna(0)

        return user_item

    def get_num_users_items(self, df):
        if df.empty:
            return 0, 0

        # Encoded IDs can be sparse within a split, so use max index + 1.
        n_users = int(df['user'].max()) + 1
        n_items = int(df['item'].max()) + 1

        return n_users, n_items