import tensorflow as tf
import numpy as np
import pandas as pd

class Widendeep:
    def __init__(self):
        self.model = tf.keras.models.load_model('my_score_model.h5')
        self.games = pd.read_csv('data/games.csv')
        self.unique_game_ids = self.games['app_id'].unique()
        self.game_id_to_index = {game_id: idx for idx, game_id in enumerate(self.unique_game_ids)}
        self.index_to_game_id = {idx: game_id for game_id, idx in self.game_id_to_index.items()}

    def ask_for_recommendation(self, user_id, n):
        all_items = np.arange(50872, dtype=np.int32)
        user_batch = np.full_like(all_items, fill_value=user_id)
        preds = self.model.predict([user_batch, all_items], verbose=0).flatten()
        top_indices = np.argsort(preds)[::-1][:n]
        result = all_items[top_indices]
        return [self.index_to_game_id[idx] for idx in result]
