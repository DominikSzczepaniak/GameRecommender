import numpy as np
import pandas as pd
import scann
from lightfm import LightFM
from tqdm import tqdm
from scipy.sparse import load_npz


class Logger:
  @staticmethod
  def log(msg):
    print(f'[LOG] {msg}')

  @staticmethod
  def error(msg):
    print(f'[ERROR] {msg}')


class LightFMscaNN:
  '''
  A hybrid recommendation system combining LightFM for matrix factorization 
  and ScaNN for fast approximate nearest neighbor search.

  Attributes:
    users (pd.DataFrame): User information from the dataset.
    games (pd.DataFrame): Game information from the dataset.
    recommendations (pd.DataFrame): Existing recommendations for validation.
    games_metadata (pd.DataFrame): Metadata for games from a JSON file.
    interactions (csr_matrix): User-game interaction matrix (sparse).
    model (LightFM): Pre-trained LightFM recommendation model.
    searcher (ScaNN searcher): ScaNN search object for approximate nearest neighbors.
    user_id_to_index (dict): Maps user IDs to matrix row indices.
    game_id_to_index (dict): Maps game IDs to matrix column indices.
    index_to_user_id (dict): Maps matrix row indices to user IDs.
    index_to_game_id (dict): Maps matrix column indices to game IDs.
    game_id_to_title (function): Maps game IDs to titles.
  '''

  def __init__(self, k=5):
    try:
      # Load datasets
      self.users = pd.read_csv('./data/users.csv')
      self.games = pd.read_csv('./data/games.csv')
      self.recommendations = pd.read_csv('./data/recommendations.csv')
      self.games_metadata = pd.read_json('./data/games_metadata.json', lines=True)

      # Load user-game interaction matrix
      self.interactions = load_npz('./data/rating_matrix_sparse.npz').tocsr()

      # Load pre-trained LightFM model
      self.model = self.loadModel()

      # Initialize ScaNN searcher for fast similarity search
      self.searcher = scann.scann_ops_pybind.builder(
        self.model.item_embeddings, k, "dot_product"
      ).score_ah(2).build()

      # Create mappings for user and game IDs
      unique_user_ids = self.users['user_id'].unique()
      unique_game_ids = self.games['app_id'].unique()
      self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
      self.game_id_to_index = {game_id: idx for idx, game_id in enumerate(unique_game_ids)}
      self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
      self.index_to_game_id = {idx: game_id for game_id, idx in self.game_id_to_index.items()}
      self.game_id_to_title = lambda game_id: self.games[self.games['app_id'] == game_id]['title'].values[0]
    except Exception as e:
      Logger.error(f'Model initialization failed: {e}')


  def fit(self, epochs=10):
    for epoch in tqdm(range(1, epochs + 1)):
      self.model.fit_partial(self.interactions, epochs=1, num_threads=20)
      try:
        # Save model embeddings and biases
        np.save('./data/model/item_embeddings.npy', self.model.item_embeddings)
        np.save('./data/model/user_embeddings.npy', self.model.user_embeddings)
        np.save('./data/model/item_biases.npy', self.model.item_biases)
        np.save('./data/model/user_biases.npy', self.model.user_biases)
      except Exception as e:
        Logger.error(f'Saving model\'s information failed: {e}')


  def loadModel(self) -> LightFM:
    model = LightFM(learning_schedule='adagrad', loss='warp')
    model.item_embeddings = np.load('./data/model/item_embeddings.npy')
    model.user_embeddings = np.load('./data/model/user_embeddings.npy')
    model.item_biases = np.load('./data/model/item_biases.npy')
    model.user_biases = np.load('./data/model/user_biases.npy')
    return model


  def list_user_liked_games(self, user_id):
    user_index = self.user_id_to_index[user_id]
    user_ratings = self.interactions[user_index].toarray()[0]

    games = []
    for idx, rating in enumerate(user_ratings):
      if rating == 1:
        games.append(self.index_to_game_id[idx])

    return games
  
  def recommend(self):
    pass
