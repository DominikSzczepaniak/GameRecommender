#TODO mappings are bugged


from scipy.sparse import load_npz, coo_matrix, vstack, save_npz
import numpy as np 
import os 
import pandas as pd
from tqdm import tqdm
import torch
import random
import joblib

class FunkSVD():
    def __init__(self, rating_matrix_path, save_dir='model_checkpoint'):
        self.save_dir = save_dir
        self.rating_matrix_path = rating_matrix_path
        self.games = None 
        self.users = None 
        self.load_mappings()
        os.makedirs(self.save_dir, exist_ok=True)
        self.load_recommendations_count()


    def recommend(self, user_id, amount):
        '''
        parameters: user_id after mapping, amount of games to recommend
        returns: list of recommended games in form of game data (app_id, name, genres, etc.)
        '''
        assert type(user_id) == int
        user_matrix, games_matrix = self.load_model()
        user_matrix = torch.tensor(user_matrix)
        games_matrix = torch.tensor(games_matrix)
        self.n_games = games_matrix.shape[0]
        self.n_users = user_matrix.shape[0]

        user_vector = user_matrix[user_id, :]
        predicted_ratings = torch.matmul(user_vector, games_matrix.T).numpy()

        played_games = set(self.get_user_history(user_id))
        all_games = set(range(self.n_games))
        non_interacted_games = list(all_games - played_games)

        score_formula = lambda item_id: predicted_ratings[item_id] #* self.score_recommendation_amount(self.count_recommendations[self.reverse_app_index[item_id]]) #* self.score_rating(self.get_game_rating(item_id))
        perfect_games = [(item_id, score_formula(item_id)) for item_id in non_interacted_games if score_formula(item_id) == 1]

        if len(perfect_games) >= amount:
            return random.choice(perfect_games, amount)
        non_interacted_ratings = [(item_id, score_formula(item_id)) for item_id in non_interacted_games]
        non_interacted_ratings.sort(key=lambda x: x[1], reverse=True)

        result = []
        for game in non_interacted_ratings:
            if len(result) == amount:
                break
            if game[0] not in played_games:
                result.append(self.get_game_data(game[0]))
        return result 
        # return non_interacted_ratings[:amount]

    

    def train(self, learning_rate=0.001, num_epochs=50, regularization = 0.1, save_freq=1, start_over=False, latent_features=10):
        '''
        Specify latent_features if start_over == True, otherwise you can ignore that
        '''
        self.load_rating_matrix()
        user_matrix, games_matrix = [], []
        if start_over:
            user_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_users, latent_features)) 
            games_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_games, latent_features)) 
        else:  
            user_matrix, games_matrix = self.load_model()
            latent_features = games_matrix.shape[1]

        for epoch in range(num_epochs):
            total_error = 0
            loop = tqdm(
                zip(self.rating_matrix_sparse.row, self.rating_matrix_sparse.col, self.rating_matrix_sparse.data),
                total=len(self.rating_matrix_sparse.data),
            )
            for user_idx, game_idx, rating in loop:
                prediction = np.dot(user_matrix[user_idx], games_matrix[game_idx])
                error = rating - prediction
                
                user_matrix[user_idx] += learning_rate * (error * games_matrix[game_idx] - regularization * user_matrix[user_idx])
                games_matrix[game_idx] += learning_rate * (error * user_matrix[user_idx] - regularization * games_matrix[game_idx])
                
                total_error += error ** 2
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Error: {total_error:.4f}, accuracy: {1 - total_error / len(self.rating_matrix_sparse.data):.4f}")
            if((epoch+1) % save_freq == 0):
                self.save_model(user_matrix, games_matrix)

    def train_one_user(self, user_id, learning_rate=0.001, num_epochs=50, regularization = 0.1, save_freq=1, start_over=False, latent_features=10):
        '''
        Train only a specific user's vector in the user matrix.
        
        Parameters:
        user_id : int - the id of the user to be trained
        learning_rate : float - the step size for gradient descent
        num_epochs : int - number of epochs to run for this user
        regularization : float - regularization parameter for weight decay
        latent_features : int - number of latent features (if starting from scratch)
        '''
        self.load_rating_matrix()
        user_idx = self.user_index[user_id]
        
        user_matrix, games_matrix = self.load_model()
        if user_matrix.shape[1] != latent_features:
            raise ValueError("Latent feature dimension mismatch between loaded model and specified value.")
        
        user_vector = user_matrix[user_idx]

        for epoch in range(num_epochs):
            total_error = 0
            user_ratings = self.rating_matrix_csr[user_idx].tocoo()

            for game_idx, rating in zip(user_ratings.col, user_ratings.data):
                prediction = np.dot(user_vector, games_matrix[game_idx])
                error = rating - prediction

                user_vector += learning_rate * (error * games_matrix[game_idx] - regularization * user_vector)
                games_matrix[game_idx] += learning_rate * (error * user_vector - regularization * games_matrix[game_idx])

                total_error += error ** 2

            print(f"User {user_id}, Epoch {epoch + 1}/{num_epochs}, Total Error: {total_error:.4f}")

        user_matrix[user_idx] = user_vector
        self.save_model(user_matrix, games_matrix)
    
    def save_model(self, user_matrix, games_matrix):
        '''
        Save user and games matrices for later loading
        '''
        np.save(os.path.join(self.save_dir, 'user_matrix.npy'), user_matrix)
        np.save(os.path.join(self.save_dir, 'games_matrix.npy'), games_matrix)

    def load_model(self):
        '''
        Load calculated user and games matrices from trained model
        '''
        user_path = os.path.join(self.save_dir, 'user_matrix.npy')
        games_path = os.path.join(self.save_dir, 'games_matrix.npy')

        if os.path.exists(user_path) and os.path.exists(games_path):
            user_matrix = np.load(user_path)
            games_matrix = np.load(games_path)
            return user_matrix, games_matrix
        else:
            raise Exception("Model files not found.")
        
    def load_mappings(self):
        '''
        Loads mappings for user_id and app_id to index and reverse mappings. 
        Saves the mappings to files if they don't exist using joblib for serialization.
        '''
        mapping_dir = 'mappings'
        user_index_file = os.path.join(mapping_dir, 'user_index.joblib')
        app_index_file = os.path.join(mapping_dir, 'app_index.joblib')
        reverse_user_index_file = os.path.join(mapping_dir, 'reverse_user_index.joblib')
        reverse_app_index_file = os.path.join(mapping_dir, 'reverse_app_index.joblib')

        os.makedirs(mapping_dir, exist_ok=True)

        if (os.path.exists(user_index_file) and 
            os.path.exists(app_index_file) and 
            os.path.exists(reverse_user_index_file) and 
            os.path.exists(reverse_app_index_file)):
            self.user_index = joblib.load(user_index_file)
            self.app_index = joblib.load(app_index_file)
            self.reverse_user_index = joblib.load(reverse_user_index_file)
            self.reverse_app_index = joblib.load(reverse_app_index_file)
        else:
            if self.games is None:
                self.games = pd.read_csv('../games.csv')
            if self.users is None:
                self.users = pd.read_csv('../users.csv')

            unique_userid = self.users['user_id'].unique()
            unique_appid = self.games['app_id'].unique()

            self.user_index = {user_id: idx for idx, user_id in enumerate(unique_userid)}
            self.app_index = {app_id: idx for idx, app_id in enumerate(unique_appid)}
            self.reverse_user_index = {idx: user_id for idx, user_id in enumerate(unique_userid)}
            self.reverse_app_index = {idx: app_id for idx, app_id in enumerate(unique_appid)}

            joblib.dump(self.user_index, user_index_file)
            joblib.dump(self.app_index, app_index_file)
            joblib.dump(self.reverse_user_index, reverse_user_index_file)
            joblib.dump(self.reverse_app_index, reverse_app_index_file)


    def load_rating_matrix(self):
        '''
        Load full 2d rating matrix of initial rating (0/1) for every user-game pair that is not null
        '''
        self.rating_matrix_sparse = load_npz(self.rating_matrix_path)
        self.rating_matrix_csr = self.rating_matrix_sparse.tocsr()
        self.n_users, self.n_games = self.rating_matrix_csr.shape 

    def load_recommendations_count(self):
        if os.path.exists('count_recommendations.npy'):
            self.count_recommendations = np.load('count_recommendations.npy', allow_pickle=True).item()
        else:
            self.count_recommendations_func()
            self.load_recommendations_count()

    def count_recommendations_func(self):
        self.recommendations = pd.read_csv('../recommendations.csv')
        self.count_recommendations = dict()
        for _, line in self.recommendations.iterrows():
            app_id = line['app_id']
            if app_id in self.count_recommendations:
                self.count_recommendations[app_id] += 1
            else:
                self.count_recommendations[app_id] = 1
        np.save('count_recommendations.npy', self.count_recommendations)

    def get_user_history(self, user_id):
        '''
        parameters: user_id after mapping
        returns: list of indices of games in V that user has played in form AFTER mapping (to get original run self.reverse_app_index[app_id])
        '''
        if not hasattr(self, 'rating_matrix_csr'):
            self.load_rating_matrix()
        played = self.rating_matrix_csr[user_id].nonzero()[1]
        return played 

    def get_game_rating(self, app_id):
        '''
        parameters: app_id before mapping
        returns: game rating - "Overwhelmingly Positive", "Very Positive", "Positive", "Mostly Positive", "Mixed", "Mostly Negative", "Negative", "Very Negative", "Overwhelmingly Negative"
        '''
        if self.games is None:
            self.games = pd.read_csv('../games.csv')
        app_id = self.app_index[app_id]
        result = self.games.loc[self.games['app_id'] == app_id, 'rating']
        if result.empty:
            return None  
        return result.iloc[0]

    def get_recommendation_amount(self, app_id):
        '''
        parameters: app_id - after mapping
        returns: amount of games to recommend
        '''
        if not hasattr(self, 'recommendations'):
            self.recommendations = pd.read_csv('../recommendations.csv')
        count = self.recommendations[self.recommendations['app_id'] == app_id].shape[0] #cache
        return count 

    def get_game_data(self, app_id):
        '''
        parameters: app_id - after mapping
        returns: game data (app_id, name, genres, etc.)
        '''
        if self.games is None:
            self.games = pd.read_csv('../games.csv')
        original_app_id = self.reverse_app_index[app_id]
        result = self.games.loc[self.games['app_id'] == original_app_id]
        if result.empty:
            return None  
        return result.iloc[0]['title']
    
    def score_game(self, user_id, app_id, score):
        '''
        parameters: user_id, app_id, score (0/1)
        returns: None
        '''
        assert score == 1 or score == 0
        if not hasattr(self, 'rating_matrix_csr'):
            self.load_rating_matrix()

        new_row_indices = [self.user_index[user_id]]
        new_col_indices = [self.app_index[app_id]]
        new_data = score 

        new_rating_matrix = coo_matrix((new_data, (new_row_indices, new_col_indices)), shape=rating_matrix_sparse.shape)

        rating_matrix_sparse = vstack([rating_matrix_sparse, new_rating_matrix])
        save_npz('rating_matrix_sparse.npz', rating_matrix_sparse)
    
    def score_rating(self, rating_name): #should be cached - app_id -> rating, takes too long
        if rating_name == 'Overwhelmingly Positive':
            return 1.0 
        elif rating_name == 'Very Positive':
            return 0.98
        elif rating_name == 'Positive':
            return 0.95
        elif rating_name == 'Mostly Positive':
            return 0.9
        elif rating_name == 'Mixed':
            return 0.85
        elif rating_name == 'Mostly Negative':
            return 0.8
        elif rating_name == 'Negative':
            return 0.7
        elif rating_name == 'Very Negative':
            return 0.6
        elif rating_name == 'Overwhelmingly Negative':
            return 0.5
        else:
            return 0 #if app_id not found or any other error just ignore the game
        
    def score_recommendation_amount(self, recommendation_count):
        if recommendation_count >= 10000:
            return 1.1
        elif recommendation_count >= 5000:
            return 1.05 
        elif recommendation_count >= 1000:
            return 1.04
        elif recommendation_count >= 750:
            return 1.02
        elif recommendation_count >= 500:
            return 1.01
        elif recommendation_count >= 300:
            return 0.98
        elif recommendation_count >= 100:
            return 0.95
        elif recommendation_count >= 50:
            return 0.9
        else:
            return 0.7

class Testing():
    def __init__(self):
        self.model = FunkSVD('../train_and_test.npz')

    def ask_for_recommendation(self, user_id, amount):
        return self.model.recommend(user_id, amount)
    
# abc = FunkSVD('../train_and_test.npz')
# abc.train(learning_rate=0.002, num_epochs=40, regularization=0.1, save_freq=1, start_over=False, latent_features=15)
print(Testing().ask_for_recommendation(13022991, 10))