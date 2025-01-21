import numpy as np
import pandas as pd
import scann
from lightfm import LightFM
from tqdm import tqdm
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split
import pickle
from scipy.sparse import coo_matrix

class Logger:
  @staticmethod
  def log(msg):
    print(f'[LOG] {msg}')

  @staticmethod
  def error(msg):
    print(f'[ERROR] {msg}')


class LightFMscaNN:
  # ----------------=[ Model initialization ]=----------------
  def __init__(self, k):
    try:
      self.users = pd.read_csv('./lightFMscaNN/data/users.csv')
      self.games = pd.read_csv('./lightFMscaNN/data/games.csv')
      self.recommendations = pd.read_csv('./lightFMscaNN/data/recommendations.csv')
      self.games_metadata = pd.read_json('./lightFMscaNN/data/games_metadata.json', lines=True)

      self.interactions = load_npz('./lightFMscaNN/data/train_and_test.npz').tocsr()
      self.load_model('bpr')

      unique_user_ids = self.users['user_id'].unique()
      unique_game_ids = self.games['app_id'].unique()

      self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
      self.game_id_to_index = {game_id: idx for idx, game_id in enumerate(unique_game_ids)}
      self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
      self.index_to_game_id = {idx: game_id for game_id, idx in self.game_id_to_index.items()}
      self.game_id_to_title = {row['app_id']: row['title'] for _, row in self.games.iterrows()}

      # Fine tuned searcher
      self.searcher = scann.scann_ops_pybind.builder(self.model.item_embeddings, k, "dot_product").score_ah(6, hash_type="lut256", training_iterations=11).build()

    except Exception as e:
      Logger.error(f'Model initialization failed: {e}')

  # ----------------=[ Model training ]=---------------
  def fit(self, loss, epochs=6):
    for epoch in tqdm(range(1, epochs + 1)):
      self.model.fit_partial(self.interactions, epochs=50, num_threads=20)

      with open(f'./lightFMscaNN/data/model/lightfm_{loss}.pkl', 'wb') as f:
        pickle.dump(self.model, f)

  def load_model(self, loss):
    with open(f'./lightFMscaNN/data/model/lightfm_{loss}.pkl', 'rb') as f:
      self.model: LightFM = pickle.load(f)


  # ----------------=[ Helper functions ]=---------------
  def list_user_liked_games(self, user_id):
    user_index = self.user_id_to_index[user_id]
    user_ratings = self.interactions[user_index].toarray()[0]

    games = []
    for idx, rating in enumerate(user_ratings):
      if rating == 1:
        games.append(self.index_to_game_id[idx])

    return games

  # -----------------=[ Prediction ]=------------------
  def embed_user(self, user_id, type=0):
    user_games = self.list_user_liked_games(user_id)

    if len(user_games) == 0:
      return np.zeros(64)
    
    if type == 0:
      game_indices = [self.game_id_to_index[game_id] for game_id in user_games]
      game_embeddings = self.model.item_embeddings[game_indices]

      user_embedding = np.mean(game_embeddings, axis=0)
      
    elif type == 1:
      user_embedding = self.model.user_embeddings[self.user_id_to_index[user_id]]
    
    return user_embedding
  

  def predict(self, user_id, k, type=0):
    user_embedding = self.embed_user(user_id, type)
    indices, scores = self.searcher.search(user_embedding)

    sorted_indices = np.argsort(-scores)
    sorted_item_indices = [indices[i] for i in sorted_indices]

    # filtered_indices = []
    # for index in sorted_item_indices:
    #   if self.index_to_game_id[index] not in user_games:
    #     filtered_indices.append(index)

    return [self.index_to_game_id[idx] for idx in sorted_item_indices]
  
  # -----------------=[ Recommendation ]=-----------------
  def ask_for_recommendation(self, user_id, k):
    return self.predict(user_id, k, 0)
  

  # -----------------=[ For Fun ]=------------------
  def similar_games(self, game_title, k):
    searcher = scann.scann_ops_pybind.builder(self.model.item_embeddings, k, "dot_product").score_ah(1).build()

    game_id = self.games[self.games['title'] == game_title]['app_id'].values[0]

    game_index = self.game_id_to_index[game_id]
    game_embedding = self.model.item_embeddings[game_index]
    indices, scores = searcher.search(game_embedding)

    return [self.index_to_game_id[idx] for idx in indices]


if __name__ == "__main__":
  model = LightFMscaNN()
  print(list(map(model.game_id_to_title, model.similar_games('ELDEN RING', 5))))
