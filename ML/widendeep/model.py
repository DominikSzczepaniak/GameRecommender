import numpy as np
import pandas as pd
import tensorflow as tf


class Widendeep:

    def __init__(self):
        self.model = tf.keras.models.load_model("my_score_model.h5")
        self.games = pd.read_csv("./data/games.csv")
        self.unique_game_ids = self.games["app_id"].unique()
        self.game_id_to_index = {
            game_id: idx
            for idx, game_id in enumerate(self.unique_game_ids)
        }
        self.index_to_game_id = {
            idx: game_id
            for game_id, idx in self.game_id_to_index.items()
        }

    def recommend(self, user_id, k):
        all_items = np.arange(50872, dtype=np.int32)
        user_batch = np.full_like(all_items, fill_value=user_id)
        preds = self.model.predict([user_batch, all_items],
                                   verbose=0).flatten()
        top_indices = np.argsort(preds)[::-1][:k]
        result = all_items[top_indices]
        return [self.index_to_game_id[idx] for idx in result]


if __name__ == "__main__":
    model = Widendeep()

    import sys

    sys.path.append("../")

    from metrics import *

    print(test_metrics(model, 20))
