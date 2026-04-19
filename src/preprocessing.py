import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

    def encode_ids(self, df):
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
        return df['user'].nunique(), df['item'].nunique()