from scipy.sparse import load_npz, coo_matrix, vstack, save_npz
import numpy as np 
import os 
import pandas as pd
from tqdm import tqdm
import torch
import random
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

class FunkSVD():
    def __init__(self, rating_matrix_path, data_directory, save_dir='model_checkpoint'):
        self.save_dir = save_dir
        self.rating_matrix_path = rating_matrix_path
        self.data_directory = data_directory
        self.games = None 
        self.users = None 
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_mappings()
        os.makedirs(os.path.join(self.base_dir, self.save_dir), exist_ok=True)
        self.load_recommendations_count()

    def recommend(self, user_id, amount):
        """
        Parameters:
        ----------
        user_id : int
            User ID after mapping.
        amount : int
            Number of games to recommend.

        Returns:
        -------
        list
            Recommended games in the form of game data (app_id, name, genres, etc.).
        """
        # Load model parameters
        user_matrix, games_matrix, user_biases, item_biases, global_mean = self.load_model()
        
        user_matrix = torch.tensor(user_matrix)
        games_matrix = torch.tensor(games_matrix)
        user_biases = torch.tensor(user_biases)
        item_biases = torch.tensor(item_biases)

        # Ensure the dimensions match
        self.n_games = games_matrix.shape[0]
        self.n_users = user_matrix.shape[0]

        # Calculate predicted ratings with biases and global mean
        user_vector = user_matrix[user_id, :]
        predicted_ratings = (
            global_mean
            + user_biases[user_id].item()
            + item_biases.numpy()
            + torch.matmul(user_vector, games_matrix.T).numpy()
        )

        # Get user's previously interacted games
        played_games = set(self.get_user_history(user_id))
        all_games = set(range(self.n_games))
        non_interacted_games = list(all_games - played_games)

        # Score formula including biases and global mean
        score_formula = lambda item_id: predicted_ratings[item_id]
        # perfect_games = [(item_id, score_formula(item_id)) for item_id in non_interacted_games if score_formula(item_id) == 1]

        # If there are enough "perfect" games, return them
        # if len(perfect_games) >= amount:
        #     return random.choice(perfect_games, amount)

        # Rank non-interacted games by predicted ratings
        non_interacted_ratings = [(item_id, score_formula(item_id)) for item_id in non_interacted_games]
        non_interacted_ratings.sort(key=lambda x: x[1], reverse=True)

        # Generate recommendation list
        result = []
        for game in non_interacted_ratings:
            if len(result) == amount:
                break
            if game[0] not in played_games:
                result.append(self.get_game_data(game[0]))

        return result


    def train(self, learning_rate=0.001, num_epochs=50, regularization=0.1, save_freq=1, start_over=False, latent_features=10):
        """
        Train the model using SGD with biases and global mean.

        Parameters:
        ----------
        learning_rate : float
            Learning rate for SGD.
        num_epochs : int
            Number of epochs to train.
        regularization : float
            L2 regularization factor.
        save_freq : int
            Frequency of saving the model.
        start_over : bool
            If True, initialize new matrices; otherwise, load saved ones.
        latent_features : int
            Number of latent features (used if start_over=True).
        """
        # Load or initialize matrices
        self.load_rating_matrix()
        if start_over:
            user_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_users, latent_features))
            games_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_games, latent_features))
            user_biases = np.zeros(self.n_users)
            item_biases = np.zeros(self.n_games)
            global_mean = np.mean(self.rating_matrix_sparse.data)
        else:
            user_matrix, games_matrix, user_biases, item_biases, global_mean = self.load_model()
            latent_features = games_matrix.shape[1]

        # Training loop
        for epoch in range(num_epochs):
            total_error = 0

            # Shuffle the data
            data = list(zip(self.rating_matrix_sparse.row, self.rating_matrix_sparse.col, self.rating_matrix_sparse.data))
            np.random.shuffle(data)

            loop = tqdm(data, total=len(data))
            for user_idx, game_idx, rating in loop:
                # Predict rating
                pred = global_mean + user_biases[user_idx] + item_biases[game_idx]
                pred += np.dot(user_matrix[user_idx], games_matrix[game_idx])

                # Calculate error
                error = rating - pred

                # Update biases
                user_biases[user_idx] += learning_rate * (error - regularization * user_biases[user_idx])
                item_biases[game_idx] += learning_rate * (error - regularization * item_biases[game_idx])

                # Update latent factors
                for factor in range(latent_features):
                    user_factor = user_matrix[user_idx, factor]
                    item_factor = games_matrix[game_idx, factor]

                    user_matrix[user_idx, factor] += learning_rate * (error * item_factor - regularization * user_factor)
                    games_matrix[game_idx, factor] += learning_rate * (error * user_factor - regularization * item_factor)

                total_error += error ** 2

            # Calculate and log metrics
            rmse = np.sqrt(total_error / len(data))
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Error: {total_error:.4f}, RMSE: {rmse:.4f}")

            # Save model periodically
            if (epoch + 1) % save_freq == 0:
                self.save_model(user_matrix, games_matrix, user_biases, item_biases, global_mean)

        print("Training complete.")


    def train_parellel(self, learning_rate=0.001, num_epochs=50, regularization=0.1, save_freq=1, start_over=False, latent_features=10):
        """
        Train the model using SGD with biases and global mean in parallel.

        Parameters:
        ----------
        learning_rate : float
            Learning rate for SGD.
        num_epochs : int
            Number of epochs to train.
        regularization : float
            L2 regularization factor.
        save_freq : int
            Frequency of saving the model.
        start_over : bool
            If True, initialize new matrices; otherwise, load saved ones.
        latent_features : int
            Number of latent features (used if start_over=True).
        """
        self.load_rating_matrix()
        if start_over:
            user_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_users, latent_features))
            games_matrix = np.random.normal(scale=1.0 / latent_features, size=(self.n_games, latent_features))
            user_biases = np.zeros(self.n_users)
            item_biases = np.zeros(self.n_games)
            global_mean = np.mean(self.rating_matrix_sparse.data)
        else:
            user_matrix, games_matrix, user_biases, item_biases, global_mean = self.load_model()
            latent_features = games_matrix.shape[1]

        def update_rating(user_idx, game_idx, rating):
            """
            Perform the SGD update for a single rating.
            """
            nonlocal total_error
            pred = global_mean + user_biases[user_idx] + item_biases[game_idx]
            pred += np.dot(user_matrix[user_idx], games_matrix[game_idx])
            error = rating - pred

            # Update biases
            user_biases[user_idx] += learning_rate * (error - regularization * user_biases[user_idx])
            item_biases[game_idx] += learning_rate * (error - regularization * item_biases[game_idx])

            # Update latent factors
            for factor in range(latent_features):
                user_factor = user_matrix[user_idx, factor]
                item_factor = games_matrix[game_idx, factor]

                user_matrix[user_idx, factor] += learning_rate * (error * item_factor - regularization * user_factor)
                games_matrix[game_idx, factor] += learning_rate * (error * user_factor - regularization * item_factor)

            return error ** 2

        for epoch in tqdm(range(num_epochs)):
            total_error = 0
            data = list(zip(self.rating_matrix_sparse.row, self.rating_matrix_sparse.col, self.rating_matrix_sparse.data))
            np.random.shuffle(data)

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(update_rating, user_idx, game_idx, rating) for user_idx, game_idx, rating in data]
                for future in as_completed(futures):
                    total_error += future.result()

            # Calculate and log metrics
            rmse = np.sqrt(total_error / len(data))
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Error: {total_error:.4f}, RMSE: {rmse:.4f}")

            # Save model periodically
            if (epoch + 1) % save_freq == 0:
                self.save_model(user_matrix, games_matrix, user_biases, item_biases, global_mean)

        print("Training complete.")



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
    
    def save_model(self, user_matrix, games_matrix, user_biases, item_biases, global_mean):
        '''
        Save user and games matrices for later loading
        '''
        np.save(os.path.join(self.base_dir, self.save_dir, 'user_matrix.npy'), user_matrix)
        np.save(os.path.join(self.base_dir, self.save_dir, 'games_matrix.npy'), games_matrix)
        np.save(os.path.join(self.base_dir, self.save_dir, 'user_biases.npy'), user_biases)
        np.save(os.path.join(self.base_dir, self.save_dir, 'item_biases.npy'), item_biases)
        np.save(os.path.join(self.base_dir, self.save_dir, 'global_mean.npy'), global_mean)


    def load_model(self):
        '''
        Load calculated user and games matrices from trained model
        '''
        user_path = os.path.join(self.base_dir, self.save_dir, 'user_matrix.npy')
        games_path = os.path.join(self.base_dir, self.save_dir, 'games_matrix.npy')
        user_biases_path = os.path.join(self.base_dir, self.save_dir, 'user_biases.npy')
        item_biases_path = os.path.join(self.base_dir, self.save_dir, 'item_biases.npy')
        global_mean_path = os.path.join(self.base_dir, self.save_dir, 'global_mean.npy')

        if os.path.exists(user_path) and os.path.exists(games_path):
            user_matrix = np.load(user_path)
            games_matrix = np.load(games_path)
            user_biases = np.load(user_biases_path)
            item_biases = np.load(item_biases_path)
            global_mean = np.load(global_mean_path)
            return user_matrix, games_matrix, user_biases, item_biases, global_mean
        else:
            raise Exception("Model files not found.")
        
    def load_mappings(self):
        '''
        Loads mappings for user_id and app_id to index and reverse mappings. 
        Saves the mappings to files if they don't exist using joblib for serialization.
        '''
        mapping_dir = 'mappings'
        user_index_file = os.path.join(self.base_dir, mapping_dir, 'user_index.joblib')
        app_index_file = os.path.join(self.base_dir, mapping_dir, 'app_index.joblib')
        reverse_user_index_file = os.path.join(self.base_dir, mapping_dir, 'reverse_user_index.joblib')
        reverse_app_index_file = os.path.join(self.base_dir, mapping_dir, 'reverse_app_index.joblib')

        os.makedirs(os.path.join(self.base_dir, mapping_dir), exist_ok=True)

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
                self.games = pd.read_csv(os.path.join(self.data_directory, 'games.csv'))
            if self.users is None:
                self.users = pd.read_csv(os.path.join(self.data_directory, 'users.csv'))

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
        file_path = os.path.join(self.base_dir, 'count_recommendations.npy')
        if os.path.exists(file_path):
            self.count_recommendations = np.load(file_path, allow_pickle=True).item()
        else:
            self.count_recommendations_func()
            self.load_recommendations_count()

    def count_recommendations_func(self):
        recommendation_file_path = os.path.join(self.data_directory, 'recommendations.csv')
        self.recommendations = pd.read_csv(recommendation_file_path)
        self.count_recommendations = dict()
        loop = tqdm(self.recommendations.iterrows(), total=self.recommendations.shape[0])
        for _, line in loop:
            app_id = line['app_id']
            if app_id in self.count_recommendations:
                self.count_recommendations[app_id] += 1
            else:
                self.count_recommendations[app_id] = 1
        file_path = os.path.join(self.base_dir, 'count_recommendations.npy')
        np.save(file_path, self.count_recommendations)

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
            self.games = pd.read_csv(os.path.join(self.data_directory, 'games.csv'))
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
            self.recommendations = pd.read_csv(os.path.join(self.data_directory, 'recommendations.csv'))
        count = self.recommendations[self.recommendations['app_id'] == app_id].shape[0] #cache
        return count 

    def get_game_data(self, app_id):
        '''
        parameters: app_id - after mapping
        returns: game data (app_id, name, genres, etc.)
        '''
        if self.games is None:
            self.games = pd.read_csv(os.path.join(self.data_directory, 'games.csv'))
        original_app_id = self.reverse_app_index[app_id]
        result = self.games.loc[self.games['app_id'] == original_app_id]
        if result.empty:
            return None  
        return result.iloc[0]
    
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
        save_npz(os.path.join(self.data_directory, 'rating_matrix_sparse.npz'), rating_matrix_sparse)
    
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
    '''
    parameters:
    data_directory: place where games.csv, recommendations.csv and users.csv are
    rating_matrix_path: path to rating_matrix_sparse.npz or equivalent
    '''
    def __init__(self, data_directory, rating_matrix_path):
        self.model = FunkSVD(rating_matrix_path, data_directory, save_dir='model_checkpoint2')

    def ask_for_recommendation(self, user_id, amount):
        return self.model.recommend2(user_id, amount)
    
# abc = FunkSVD('../train_and_test.npz')
# abc.train(learning_rate=0.002, num_epochs=40, regularization=0.1, save_freq=1, start_over=False, latent_features=15)
if __name__ == '__main__':
    # print(Testing('data', 'data/train_and_test.npz').ask_for_recommendation(13022991, 10))
    FunkSVD('data/train_and_test.npz', 'data', save_dir='model_checkpoint5').train_parellel(learning_rate=0.01, num_epochs=10, regularization=0.005, save_freq=1, start_over=False, latent_features=30)
    