import pandas as pd
from scipy.sparse import load_npz


class HistoryGetter:
  def __init__(self, model_folder_path: str):
    self.rest_history = load_npz(model_folder_path + '/data/rest_test.npz')
    self.user_histories = {}  # Dictionary to store user histories
    self.path = model_folder_path
    self.load_mappings()


  def get_user_actual(self, userId: int):
    user_row = self.rest_history.row
    user_col = self.rest_history.col

    # Filter entries based on the user ID
    user_entries = user_col[user_row == userId]

    # Map column indices to app_ids using reverse_app_index
    user_actual = [self.reverse_app_index[col_index] for col_index in user_entries]

    return user_actual
    

  def load_mappings(self):
    self.users = pd.read_csv(self.path + '/data/users.csv')
    self.games = pd.read_csv(self.path + '/data/games.csv')

    unique_user_ids = self.users['user_id'].unique()
    unique_game_ids = self.games['app_id'].unique()

    self.user_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    self.app_index = {game_id: idx for idx, game_id in enumerate(unique_game_ids)}
    self.reverse_user_index = {idx: user_id for user_id, idx in self.user_index.items()}
    self.reverse_app_index = {idx: game_id for game_id, idx in self.app_index.items()}


  def get_user_test_ids(self):
    matrix = load_npz(self.path + '/data/test_matrix.npz')
    user_ids = set(matrix.row)  
    return user_ids


  def load_all_items_id(self):
    items_df = pd.read_csv(self.path + '/data/games.csv')

    all_items = items_df['app_id'].unique()
    return all_items
