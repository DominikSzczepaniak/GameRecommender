import numpy as np 
import pandas as pd
from surprise import Reader, Dataset
import surprise
from scipy.sparse import load_npz 
import joblib
import time 


class SVDppCustom:
    def __init__(self, data_path):
        self.data_path = data_path

    def train(self):
        print("Loading data..")
        data = self.load_data(self.data_path)
        print("Data loaded..")
        print("Building trainset..")
        trainset = data.build_full_trainset()
        print("Trainset built..")
        algo = surprise.prediction_algorithms.matrix_factorization.SVDpp(n_factors=128, n_epochs=20, cache_ratings=True, verbose=True)
        print("Training algo")
        algo.fit(trainset)
        joblib.dump(algo, 'model')

    def load_data(self, data_path):
        df = self.convert_sparse_to_csv(data_path)
        reader = Reader(rating_scale=(0,1))
        data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
        return data 

    def convert_sparse_to_csv(self, path):
        sparsed = load_npz(path)
        rows = sparsed.row
        columns = sparsed.col
        data = sparsed.data
        recommendations = pd.DataFrame({'userID': rows, 'itemID': columns, 'rating': data})
        return recommendations
    
    def recommend(self, user_id, k):
        """Generates top_k recommendations for user_id using SVD++

        Args:
            user_id (_type_): id of user for games recommendations
            k (_type_): how many games to return
        """
        # Load the trained model
        if not hasattr(self, 'algo'):
            self.algo = joblib.load('model')
            self.most_popular = [47380, 13173, 12800, 47791, 14398, 16071, 12711, 13176, 47538, 15273, 14163, 13181,
                14535, 13273, 50781, 47474, 12712, 47637, 11718, 15719, 14434, 15363, 47760, 14095,
                47620, 47689, 14098, 12744, 47607, 13035, 12489, 14457, 12717, 15364, 50787, 15040,
                15278, 47793, 15103, 14563, 50782, 47494, 13503, 12740, 15727, 13805, 47890, 47660,
                14164, 14375, 13504, 15276, 15358, 11968, 47730, 48122, 14453, 47377, 14777, 14432,
                14376, 27666, 48013, 13505, 47653, 15244, 12573, 14166, 47582, 47545, 47470, 14396,
                13274, 14461, 14778, 13598, 15389, 15281, 47531, 48514, 47759, 15785, 12658, 13568,
                47407, 47281, 13803, 12689, 13909, 48353, 15401, 15097, 13912, 47548, 48299, 14459,
                480, 47405, 47648, 47649]
        
        try:
            # Convert raw user id to inner id
            inner_uid = self.algo.trainset.to_inner_uid(user_id)
        except ValueError:
            # print(user_id)
            # User not found in the training set
            return self.most_popular[:k]
        
        # Get all item inner IDs the user has rated
        rated_items = {iid for (iid, _) in self.algo.trainset.ur[inner_uid]}
        
        # Collect all predictions for items not rated by the user
        predictions = []
        for inner_iid in self.algo.trainset.all_items():
            if inner_iid not in rated_items:
                raw_iid = self.algo.trainset.to_raw_iid(inner_iid)
                predicted_rating = self.algo.predict(user_id, raw_iid).est
                predictions.append((raw_iid, predicted_rating))
        
        # Sort the predictions to get top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k = [item_id for item_id, _ in predictions[:k]]
        
        return top_k

if __name__ == "__main__":
    abc = SVDppCustom('./data/train_and_test.npz') 
    print(abc.recommend(1, 10))
    times = time.time()
    print(abc.recommend(2, 10))
    print(f"{time.time() - times:.2f}s")