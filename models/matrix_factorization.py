import numpy as np

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=50, epochs=30, lr=0.01, reg=0.02, decay_rate=0.0, time_col="timestamp"):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.decay_rate = decay_rate
        self.time_col = time_col

    def _get_time_weights(self, df):
        if self.decay_rate <= 0 or self.time_col not in df.columns:
            return np.ones(len(df), dtype=float)

        timestamps = df[self.time_col].astype(float).to_numpy()
        valid_mask = np.isfinite(timestamps)

        # If all timestamps are invalid/missing, disable decay for this batch.
        if not np.any(valid_mask):
            return np.ones(len(df), dtype=float)

        max_timestamp = np.nanmax(timestamps)
        # Replace invalid timestamps with max timestamp so they get neutral weight 1.0.
        safe_timestamps = np.where(valid_mask, timestamps, max_timestamp)
        age_days = (max_timestamp - safe_timestamps) / (60 * 60 * 24)
        age_days = np.clip(age_days, a_min=0.0, a_max=None)

        return np.exp(-self.decay_rate * age_days)

    def fit(self, df):
        # init embeddings
        self.U = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
        self.V = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors))

        # bias terms 🔥
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        # global mean
        self.global_mean = df['rating'].mean()
        time_weights = self._get_time_weights(df)

        # SGD training
        for epoch in range(self.epochs):
            loss = 0
            weight_sum = 0.0

            for idx, row in enumerate(df.itertuples()):
                u = row.user
                i = row.item
                r = row.rating
                weight = time_weights[idx]

                if not np.isfinite(weight) or not np.isfinite(r):
                    continue

                pred = self.predict(u, i)
                err = r - pred

                if not np.isfinite(err):
                    continue

                weight_sum += weight

                # update biases
                self.user_bias[u] += self.lr * weight * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * weight * (err - self.reg * self.item_bias[i])

                # update embeddings
                self.U[u] += self.lr * weight * (err * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * weight * (err * self.U[u] - self.reg * self.V[i])

                loss += weight * err**2

            mean_loss = loss / max(weight_sum, 1e-12)
            print(
                f"Epoch {epoch+1}/{self.epochs}, "
                f"WeightedLoss: {loss:.2f}, MeanWeightedMSE: {mean_loss:.4f}"
            )

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