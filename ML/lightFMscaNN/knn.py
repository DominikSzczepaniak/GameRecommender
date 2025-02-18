import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors


class KNNRecommender:
    def __init__(self):
        self.interaction = load_npz('./data/train_and_test.npz').tocsr()
        self.users = pd.read_csv('./data/users.csv')
        self._train_knn()
        unique_user_ids = self.users["user_id"].unique()
        self.user_id_to_index = {
                user_id: idx
                for idx, user_id in enumerate(unique_user_ids)
            }
        
    def _train_knn(self):
        """Trains the KNN model on the sparse matrix."""
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=21)
        self.knn_model.fit(self.interaction)
        
    def recommend(self, user_id, k=10):
        user_vector = self.interaction[self.user_id_to_index[user_id]]

        scores, indices = self.knn_model.kneighbors(user_vector, n_neighbors=k+1)
        scores = scores[0]
        indices = indices[0]

        sorted_indices = np.argsort(-scores)
        sorted_item_indices = [indices[i] for i in sorted_indices]

        return sorted_item_indices[:k]
    

if __name__ == '__main__':
    import sys

    sys.path.append("../")

    from metrics import *

    print(test_metrics(KNNRecommender(), 20))
