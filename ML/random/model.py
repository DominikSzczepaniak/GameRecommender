import numpy as np
import pandas as pd


class randomModel:

    def __init__(self):
        self.games = pd.read_csv("./data/games.csv")
        self.rows = range(0, len(self.games) + 1)

    def recommend(self, user_id, k):
        return np.random.choice(self.rows, k, replace=False).tolist()


if __name__ == "__main__":
    model = randomModel()

    import sys

    sys.path.append("../")

    from other_models.metrics import *

    stats = []
    for _ in range(10):
        stats.append(test_metrics(model, 10))

    mean_values = {
        key: np.mean([d[key] for d in stats])
        for key in stats[0].keys()
    }

    print(mean_values)
