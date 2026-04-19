import numpy as np
import pandas as pd


class BaselineModel:
    def __init__(self):
        self.global_mean = None
        self.user_mean = None
        self.item_mean = None
        self.user_bias = None
        self.item_bias = None

    def fit(self, df):
        """
        Train baseline models
        """
        # Global mean
        self.global_mean = df['rating'].mean()

        # User mean
        self.user_mean = df.groupby('user')['rating'].mean()

        # Item mean
        self.item_mean = df.groupby('item')['rating'].mean()

        # Bias model
        self.user_bias = self.user_mean - self.global_mean
        self.item_bias = self.item_mean - self.global_mean

    def predict_global(self):
        return self.global_mean

    def predict_user(self, user):
        return self.user_mean.get(user, self.global_mean)

    def predict_item(self, item):
        return self.item_mean.get(item, self.global_mean)

    def predict_bias(self, user, item):
        """
        Bias model:
        r_hat = global + user_bias + item_bias
        """
        u_bias = self.user_bias.get(user, 0)
        i_bias = self.item_bias.get(item, 0)

        return self.global_mean + u_bias + i_bias