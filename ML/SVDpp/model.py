import numpy as np 
import pandas as pd
from surprise import Reader, Dataset
import surprise
from scipy.sparse import load_npz 
import joblib


class SVDppCustom:
    def __init__(self, data_path):
        self.data_path = data_path

    def train(self):
        print("Loading data..")
        data = self.load_data(self.data_path)
        print("Data loaded..")
        print("Building trainset..")
        data.build_full_trainset()
        print("Trainset built..")
        algo = surprise.prediction_algorithms.matrix_factorization.SVDpp(n_factors=128, n_epochs=20, cache_ratings=True, verbose=True)
        print("Training algo")
        algo.fit(data)
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

if __name__ == "__main__":
    SVDppCustom('../funksvd/data/train_and_test.npz').train()