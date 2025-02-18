from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required
import lightfm
from lightfm import LightFM
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import pickle
from typing import List
import scann


class Logger:
    @staticmethod
    def log(msg):
        print(f"[LOG] {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}")


app = Flask(__name__)

class Logger:
  @staticmethod
  def log(msg):
    print(f'[LOG] {msg}')

  @staticmethod
  def error(msg):
    print(f'[ERROR] {msg}')


class Recommender:
  def __init__(self, k):
    try:
      with open('./data/model/lightfm_warp64.pkl', 'rb') as f:
        self.model: LightFM = pickle.load(f)

      self.games = pd.read_csv('./data/games.csv')
      self.users = pd.read_csv('./data/users.csv')
      self.interactions = load_npz('./data/train_and_test.npz').tocsr()
      self.recommendations = pd.read_csv('./data/recommendations2.csv')

      self.unique_user_ids = self.users['user_id'].unique()
      self.unique_game_ids = self.games['app_id'].unique()

      self.user_index = {user_id: idx for idx, user_id in enumerate(self.unique_user_ids)}
      self.app_index = {game_id: idx for idx, game_id in enumerate(self.unique_game_ids)}
      self.reverse_user_index = {idx: user_id for user_id, idx in self.user_index.items()}
      self.reverse_app_index = {idx: game_id for game_id, idx in self.app_index.items()}
    except Exception as e:
      Logger.error(f'Error loading model: {e}')
      # We do wanna raise the exception here...
      raise e
  

  def similar_games(self, game_title, k):
    searcher = scann.scann_ops_pybind.builder(self.model.item_embeddings, k, "dot_product").score_ah(6, hash_type="lut256", training_iterations=11).build()
    game_id = self.games[self.games['title'] == game_title]['app_id'].values[0]

    game_index = self.app_index[game_id]
    game_embedding = self.model.item_embeddings[game_index]
    indices, _ = searcher.search(game_embedding)

    return [self.reverse_app_index[idx] for idx in indices]


  def embed_user(self, user_id: int):
    # TODO: fetch user games from database
    user_games = None

    if len(user_games) == 0:
      return np.zeros(64)
    
    game_indices = [self.app_index[game_id] for game_id in user_games]
    game_embeddings = self.model.item_embeddings[game_indices]

    user_embedding = np.mean(game_embeddings, axis=0)
    
    return user_embedding
  

  def predict(self, user_id: int, k: int):
    searcher = scann.scann_ops_pybind.builder(self.model.item_embeddings, k, "dot_product").score_ah(6, hash_type="lut256", training_iterations=11).build()
    user_embedding = self.embed_user(user_id)
    indices, _ = searcher.search(user_embedding)

    return [self.reverse_app_index[idx] for idx in indices]
  

class modelAPI:
  def __init__(self):
    self.model = Recommender()
    app.config["JWT_SECRET_KEY"] = "xpp"  # !!!!!!!!!!!!!!
    self.jwt = JWTManager(app)


  @app.route('/learnUser', methods=['POST'])
  @jwt_required()
  def learn(self):
    '''input: 
      [ { UserId, AppId, Opinion, Playtime }, ... ]
    '''

    data = request.get_json()

    if not isinstance(data, list):
      Logger.error('in learn: Invalid input')
      return jsonify({'error': 'Invalid input'}), 400

    userId = int(data[0]['UserId'])
    played_games = []
    hours = []
    recommended = []

    for entry in data:
      played_games.append(int(entry['AppId']))
      hours.append(int(entry['Playtime']))
      recommended.append(int(entry['Opinion']))

    # TODO: Add the user to the database

    return jsonify({}), 200
    

  @app.route('/recommendations', methods=['POST'])
  @jwt_required()
  def recommend(self):
    '''input: 
      { UserId, k }

      output:
      [ AppId, ... ]
    '''

    data = request.get_json()

    if not isinstance(data, dict):
      Logger.error('in recommend: Invalid input')
      return jsonify({'error': 'Invalid input'}), 400

    user_id = int(data['UserId'])
    k = int(data['k'])

    recommendations = self.model.predict(user_id, k)

    return jsonify(recommendations), 200


if __name__ == '__main__':
  api = modelAPI()
  app.run(port=5000)
