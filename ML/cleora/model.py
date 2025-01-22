
from pycleora import SparseMatrix
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import joblib
from sklearn.neighbors import NearestNeighbors
import sys

class Cleora:
    def convert_sparse_to_csv(self, path):
        sparsed = load_npz(path)
        rows = sparsed.row
        columns = sparsed.col
        data = sparsed.data
        recommendations = pd.DataFrame({'user_id': rows, 'app_id': columns, 'rating': data})
        return recommendations

    def fit(self):
        users_game = self.convert_sparse_to_csv('./data/train_and_test.npz').groupby('user_id')['app_id'].apply(list).values

        cleora_input = map(lambda x: ' '.join(map(str, x)), users_game)

        mat = SparseMatrix.from_iterator(cleora_input, columns='complex::reflexive::app_id')
        embedding_dim = 4096
        embeddings = mat.initialize_deterministically(embedding_dim)

        NUM_WALKS = 2
        for i in range(NUM_WALKS):
            embeddings = mat.left_markov_propagate(embeddings)
            embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)

        joblib.dump(embeddings, 'game_embeddings.joblib')
        joblib.dump(mat.entity_ids, 'game_entity_ids.joblib')

        knn = NearestNeighbors(metric='cosine', algorithm='auto')
        knn.fit(embeddings)
        joblib.dump(knn, 'knn_model.joblib')

    def recommend_similar_games(self, game_id, top_n=5):
        if not hasattr(self, 'embeddings'):
            self.embeddings = joblib.load('./cleora/game_embeddings.joblib')
            self.entity_ids = joblib.load('./cleora/game_entity_ids.joblib')
            self.knn = joblib.load('./cleora/knn_model.joblib')
        game_id = str(game_id)
        if game_id not in self.entity_ids:
            raise ValueError(f"Game ID '{game_id}' not found.")
        
        game_index = self.entity_ids.index(game_id)
        game_embedding = self.embeddings[game_index].reshape(1, -1)
        
        distances, indices = self.knn.kneighbors(game_embedding, n_neighbors=top_n + 1)
        similar_games = [self.entity_ids[i] for i in indices.flatten() if i != game_index]
        
        return similar_games, distances.flatten()
    
    def recommend_for_game_history(self, game_history, k=10):
        '''
        parameters: game_history - history of games user played
                    k - number of games to recommend
        returns: list of recommended games
        '''
        recommended = {}
        if len(game_history) == 0:
            return self.recommend_similar_games(730, top_n=k)[0]
        for game in game_history:
            try:
                similar_games, distances = self.recommend_similar_games(game, top_n=k)
                for i, game in enumerate(similar_games):
                    if game not in recommended:
                        recommended[game] = distances[i]
            except ValueError as e:
                print(e)
                continue
        recommended = dict(sorted(recommended.items(), key=lambda x: x[1]/len(game_history)))
        return list(recommended.keys())[:k]

    def ask_for_recommendation(self, user_id, k=10):
        '''
        parameters: user_id - user id
                    k - number of games to recommend
        returns: list of recommended games
        '''
        if not hasattr(self, 'recommendations'):
            self.recommendations = self.convert_sparse_to_csv('./cleora/data/train_and_test.npz')
        user_history = self.recommendations.loc[self.recommendations['user_id'] == user_id]['app_id'].values
        return self.recommend_for_game_history(user_history, k=k)

if __name__ == '__main__':
    # games = pd.read_csv('games.csv')
    # sys.stdout.reconfigure(encoding='utf-8')
    # def get_game_data(app_id):
    #     '''
    #     parameters: app_id - after mapping
    #     returns: game data (app_id, name, genres, etc.)
    #     '''
    #     app_id = int(app_id)
    #     result = games.loc[games['app_id'] == app_id]
    #     if result.empty:
    #         return None  
    #     return result.iloc[0]['title']
    
    Cleora().fit()

    # cr = CleoraRecommender()
    # game_history = [730, 280, 379720]
    # print(f"Dla gier: {list(map(get_game_data, game_history))}\n Polecone zostaly:")
    # print(list(map(get_game_data, recommend_for_game_history([730, 280, 379720], k=10, recommender=cr))))


