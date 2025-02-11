import numpy as np
import pandas as pd
from funk_svd import SVD
from scipy.sparse import load_npz
import os
import joblib

class FunkSVD:
    def __init__(self, train_data_path, test_data_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.df = self.convert_sparse_to_csv(train_data_path)
        self.test_df = self.convert_sparse_to_csv(test_data_path)
        self.n_items = len(self.df['i_id'].unique())
        self.all_item_ids = [i for i in range(1, self.n_items+1)]
        self.save_dir = save_dir
        self.loaded = False

    def train(self, lr=0.01, n_features=100, n_epochs=30, regularization=0.02):
        self.model = SVD(lr=lr, reg=regularization, n_factors=n_features, n_epochs=n_epochs, min_rating = 0, max_rating = 1, shuffle=True)
        self.model.fit(self.df, self.test_df)
        self.save_model()

    def load_model(self):
        assert os.path.exists(os.path.join(self.save_dir, 'pu_.npy'))
        print("Loading model...")
        self.model = SVD(lr=0.001, reg=0.02, n_factors=100, n_epochs=10, min_rating = 0, max_rating = 1, shuffle=True)
        self.model.pu_ = joblib.load(os.path.join(self.save_dir, 'pu_.joblib'))
        self.model.qi_ = joblib.load(os.path.join(self.save_dir, 'qi_.joblib'))
        self.model.bu_ = joblib.load(os.path.join(self.save_dir, 'bu_.joblib'))
        self.model.bi_ = joblib.load(os.path.join(self.save_dir, 'bi_.joblib'))
        self.model.item_mapping_ = joblib.load(os.path.join(self.save_dir, 'itemmappings.joblib'))
        self.model.user_mapping_ = joblib.load(os.path.join(self.save_dir, 'usermappings.joblib'))
        self.model.global_mean_ = self.df['rating'].mean()
        print("Model loaded")

    def save_model(self):
        joblib.dump(self.model.pu_, os.path.join(self.save_dir, 'pu_.joblib'))
        joblib.dump(self.model.qi_, os.path.join(self.save_dir, 'qi_.joblib'))
        joblib.dump(self.model.bu_, os.path.join(self.save_dir, 'bu_.joblib'))
        joblib.dump(self.model.bi_, os.path.join(self.save_dir, 'bi_.joblib'))
        joblib.dump(self.model.item_mapping_, os.path.join(self.save_dir, 'itemmappings.joblib'))
        joblib.dump(self.model.user_mapping_, os.path.join(self.save_dir, 'usermappings.joblib'))

    def load_mappings(self):
        print("Loading mappings")
        games = pd.read_csv('./data/games.csv')
        gameIds = games['app_id'].unique()
        self.mapGameId = {game_id: idx for idx, game_id in enumerate(gameIds)}
        self.mapGameIndex = {idx: game_id for game_id, idx in self.mapGameId.items()}
        print("Mappings loaded")

    def convert_sparse_to_csv(self, path):
        sparsed = load_npz(path)
        rows = sparsed.row
        columns = sparsed.col
        data = sparsed.data
        recommendations = pd.DataFrame({'u_id': rows, 'i_id': columns, 'rating': data})
        return recommendations

    def recommend(self, user_id, k):
        if not hasattr(self, 'model') or not self.loaded:
            self.load_model()
            self.loaded = True 
            self.load_mappings()
        rated_items = self.df[(self.df['u_id'] == user_id) & (self.df['rating'] == 1)]['i_id'].unique()
        
        candidate_items = list(set(self.all_item_ids) - set(rated_items))
        
        if not candidate_items:
            return []  
        
        user_items = pd.DataFrame({
            'u_id': [user_id] * len(candidate_items),
            'i_id': candidate_items
        })
        
        predictions = self.model.predict(user_items)
        
        user_items['predicted_rating'] = predictions
        top_k = user_items.sort_values('predicted_rating', ascending=False).head(k)
        
        result = top_k['i_id'].tolist()
        return result 
    
if __name__ == "__main__":
    FunkSVD("./data/train_and_test.npz", "./data/rest_test.npz", "model_github_100_features_second_try").train(n_epochs=50)
    