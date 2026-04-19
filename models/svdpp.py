import numpy as np
from collections import defaultdict


class SVDPP:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.02, epochs=5):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        self.U = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
        self.V = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))
        self.Y = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))  # implicit

        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        self.global_mean = 0

        self.user_items = defaultdict(list)

    def fit(self, df):
        self.global_mean = df['rating'].mean()

        # Build implicit feedback
        for row in df.itertuples():
            self.user_items[row.user].append(row.item)

        for epoch in range(self.epochs):
            total_loss = 0

            for row in df.itertuples():
                u = row.user
                i = row.item
                r = row.rating

                items_u = self.user_items[u]
                sqrt_Nu = np.sqrt(len(items_u)) if items_u else 1

                # implicit sum
                y_sum = np.sum(self.Y[items_u], axis=0) / sqrt_Nu if items_u else 0

                pred = (self.global_mean +
                        self.user_bias[u] +
                        self.item_bias[i] +
                        np.dot(self.V[i], self.U[u] + y_sum))

                err = r - pred

                # update biases
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                # update embeddings
                self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (err * (self.U[u] + y_sum) - self.reg * self.V[i])

                # update implicit factors
                for j in items_u:
                    self.Y[j] += self.lr * (err * self.V[i] / sqrt_Nu - self.reg * self.Y[j])

                total_loss += err**2

            print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    def predict(self, u, i):
        items_u = self.user_items[u]
        sqrt_Nu = np.sqrt(len(items_u)) if items_u else 1

        y_sum = np.sum(self.Y[items_u], axis=0) / sqrt_Nu if items_u else 0

        return (self.global_mean +
                self.user_bias[u] +
                self.item_bias[i] +
                np.dot(self.V[i], self.U[u] + y_sum))