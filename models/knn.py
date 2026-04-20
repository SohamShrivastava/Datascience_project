import numpy as np
from collections import defaultdict


class KNNModel:
    def __init__(self, k=20, min_common=3):
        self.k = k
        self.min_common = min_common
        self.user_ratings = {}
        self.global_mean = 0.0
        self.user_means = {}
        self.item_users = defaultdict(set)

    def fit(self, df):
        self.global_mean = df['rating'].mean()

        # Build user → {item: rating}
        for row in df.itertuples():
            self.user_ratings.setdefault(row.user, {})[row.item] = row.rating

        # Compute per-user mean (was commented out — BUG FIX)
        for user, ratings in self.user_ratings.items():
            self.user_means[user] = np.mean(list(ratings.values()))

        # Build item → users index for fast lookup
        for user, ratings in self.user_ratings.items():
            for item in ratings:
                self.item_users[item].add(user)

    def _similarity(self, u1, u2):
        r1 = self.user_ratings.get(u1, {})
        r2 = self.user_ratings.get(u2, {})

        common = set(r1.keys()) & set(r2.keys())
        if len(common) < self.min_common:
            return 0.0

        mean1 = self.user_means[u1]
        mean2 = self.user_means[u2]

        num = sum((r1[i] - mean1) * (r2[i] - mean2) for i in common)
        den = (
            np.sqrt(sum((r1[i] - mean1) ** 2 for i in common)) *
            np.sqrt(sum((r2[i] - mean2) ** 2 for i in common))
        )
        return num / den if den > 0 else 0.0

    def predict(self, user, item):
        if user not in self.user_ratings:
            return self.global_mean

        # Fast lookup via item_users index
        candidates = list(self.item_users.get(item, set()) - {user})

        if not candidates:
            return self.user_means.get(user, self.global_mean)

        sims = [(u, self._similarity(user, u)) for u in candidates]
        sims = sorted(sims, key=lambda x: abs(x[1]), reverse=True)[:self.k]
        sims = [(u, s) for u, s in sims if s > 0]

        if not sims:
            return self.user_means.get(user, self.global_mean)

        user_mean = self.user_means.get(user, self.global_mean)
        num = sum(s * (self.user_ratings[u][item] - self.user_means[u])
                  for u, s in sims)
        den = sum(abs(s) for _, s in sims)

        return np.clip(user_mean + num / den, 0.5, 5.0)

    def get_neighbor_items(self, user, top_n=200):
        if user not in self.user_ratings:
            return []

        # Only check users who share items (faster than all users)
        user_items = set(self.user_ratings[user].keys())
        candidate_users = set()
        for item in user_items:
            candidate_users.update(self.item_users[item])
        candidate_users.discard(user)

        sims = [(u, self._similarity(user, u)) for u in candidate_users]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:self.k]

        neighbor_items = set()
        for u, s in sims:
            if s > 0:
                neighbor_items.update(self.user_ratings[u].keys())

        train_items = set(self.user_ratings[user].keys())
        return list(neighbor_items - train_items)[:top_n]