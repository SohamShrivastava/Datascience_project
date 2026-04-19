import numpy as np

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=50, epochs=30, lr=0.01, reg=0.02):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg

    def fit(self, df):
        # init embeddings
        self.U = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
        self.V = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors))

        # bias terms 🔥
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        # global mean
        self.global_mean = df['rating'].mean()

        # SGD training
        for epoch in range(self.epochs):
            loss = 0

            for row in df.itertuples():
                u = row.user
                i = row.item
                r = row.rating

                pred = self.predict(u, i)
                err = r - pred

                # update biases
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                # update embeddings
                self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (err * self.U[u] - self.reg * self.V[i])

                loss += err**2

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.2f}")

    def predict(self, user, item):

        user_bias = getattr(self, "user_bias", np.zeros(self.n_users))
        item_bias = getattr(self, "item_bias", np.zeros(self.n_items))

        pred = (
            self.global_mean
            + user_bias[user]
            + item_bias[item]
            + np.dot(self.U[user], self.V[item])
        )

        return np.clip(pred, 0.5, 5)