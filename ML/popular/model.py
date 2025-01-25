import numpy as np
from scipy.sparse import load_npz


class popularModel:

    def __init__(self):
        self.interactions = load_npz("./data/train_and_test.npz").tocsr()

        train_popularity = np.array(self.interactions.sum(axis=0)).flatten()

        train_popularity += 1

        log_popularity = np.log(train_popularity)
        self.popularity_weights = (log_popularity - log_popularity.min()) / (
            log_popularity.max() - log_popularity.min())

    def recommend(self, user_id, k):
        _, known_items = self.interactions[user_id].nonzero()

        all_items = np.arange(self.interactions.shape[1])
        candidate_items = np.setdiff1d(all_items, known_items)

        pop_scores = self.popularity_weights[candidate_items]

        top_k_indices = np.argsort(-pop_scores)[:k]

        return candidate_items[top_k_indices]


if __name__ == "__main__":
    model = popularModel()

    import sys

    sys.path.append("../")

    from metrics import *

    print(test_metrics(model, 50))
