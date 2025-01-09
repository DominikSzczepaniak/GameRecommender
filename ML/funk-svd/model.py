from scipy.sparse import load_npz
import numpy as np 
import os 
import pandas as pd

class FunkSVD():
    def __init__(self, n_games, n_users, save_dir='model_checkpoint'):
        self.save_dir = save_dir
        self.n_games = n_games 
        self.n_users = n_users
        os.makedirs(self.save_dir, exist_ok=True)
    
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
            raise Exception()

    def load_rating_matrix(self, rating_matrix_path):
        '''
        Load full 2d rating matrix of initial rating (0/1) for every user-game pair that is not null
        '''
        self.rating_matrix_sparse = load_npz(rating_matrix_path)
        self.rating_matrix_csr = rating_matrix_path.tocsr()
        self.n_users, self.n_games = self.rating_matrix_csr.shape 

    def train(self, learning_rate=0.001, num_epochs=50, save_freq=1, start_over=False, latent_features=10):
        '''
        Specify latent_features if start_over == True, otherwise you can ignore that
        '''
        self.load_rating_matrix()
        regularization = 0.1      # Regularization term to prevent overfitting
        user_matrix, games_matrix = [], []
        if start_over:
            user_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_users, latent_features)) 
            games_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_games, latent_features)) 
        else:  
            user_matrix, games_matrix = self.load_model()
            latent_features = games_matrix.shape[1]

        for epoch in range(num_epochs):
            total_error = 0
            for user_idx, game_idx, rating in zip(self.rating_matrix_sparse.row, self.rating_matrix_sparse.col, self.rating_matrix_sparse.data):
                prediction = np.dot(user_matrix[user_idx], games_matrix[game_idx])
                error = rating - prediction
                
                user_matrix[user_idx] += learning_rate * (error * games_matrix[game_idx] - regularization * user_matrix[user_idx])
                games_matrix[game_idx] += learning_rate * (error * user_matrix[user_idx] - regularization * games_matrix[game_idx])
                
                total_error += error ** 2
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Error: {total_error:.4f}")
            if((epoch+1) % save_freq == 0):
                self.save_model(user_matrix, games_matrix)

    def score_rating(self, rating_name):
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

    def get_user_history(self, user_id):
        '''
        parameters: user_id
        returns: list of indices of games in V that user has played
        '''
        pass #TODO api call

    def get_game_rating(self, app_id):
        games = pd.read_csv('games.csv')
        result = games.loc[games['app_id'] == app_id, 'rating']
        if result.empty:
            return None  
        return result.iloc[0]

    def recommend(self, user_id, amount):
        assert type(user_id) == int
        user_matrix, games_matrix = self.load_model()

        if user_id not in user_matrix: #??
            raise Exception 

        user_vector = user_matrix[user_id, :]
        predicted_ratings = np.dot(user_vector, games_matrix)

        played_games = set(get_user_history(user_id))
        all_games = set(range(self.n_games))
        non_interacted_games = list(all_games - played_games)

        score_formula = lambda item_id: predicted_ratings[item_id] * self.score_rating(self.get_game_rating(item_id))

        non_interacted_ratings = [(item_id, score_formula(item_id)) for item_id in non_interacted_items]

        non_interacted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return non_interacted_ratings[:top_k]
