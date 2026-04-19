import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = len([r for r in recommended_k if r in relevant_set])
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    if len(relevant_set) == 0:
        return 0

    hits = len([r for r in recommended_k if r in relevant_set])
    return hits / len(relevant_set)