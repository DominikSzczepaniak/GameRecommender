# from recmetrics import novelty, prediction_coverage, catalog_coverage, _ark, _apk, mark, mapk, personalization
from typing import List
import numpy as np
from scipy.sparse import load_npz, coo_matrix
#diversity = catalog_coverage 
#novelty = novelty
#coverage popularity = prediction coverage


def NDCG(predicted: List[List], actual: List[List], k: int):
    '''source: https://en.wikipedia.org/wiki/Discounted_cumulative_gain'''
    
    def ndcg_at(predicted_at: List[int], actual_at: List[int]):
        relevance = [1 if game in actual_at else 0 for game in predicted_at[:k]]
        ideal_relevance = sorted(relevance)

        DCG = np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
        ideal_DCG = np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
        epsilon = 1e-15
        
        return DCG / ideal_DCG if ideal_DCG > 0 else 0
    
    ndcg_scores = [ndcg_at(predicted[i], actual[i]) for i in range(len(predicted))]

    return np.mean(ndcg_scores)
    

def test_model(predicted: List[List], actual: List[List], k: int, user_amount: int, recommendation_size: int, all_items: List):
    novelty_score = novelty(predicted, actual)
    prediction_coverage_score = prediction_coverage(predicted, actual)
    catalog_coverage_score = catalog_coverage(predicted, all_items, k)
    personalization_score = personalization(predicted)
    NDCG_score = NDCG(predicted, actual, k) #to change if needed 
    ark_score = _ark(predicted, actual, k)
    MAP_score = mapk(predicted, actual, k)
    return {"novelty": novelty_score, "prediction_coverage": prediction_coverage_score, "catalog_coverage": catalog_coverage_score, "personalization": personalization_score, "NDCG": NDCG_score, "ark": ark_score, "MAP": MAP_score}

class HistoryGetter:
    def __init__(self, rest_history_file_path: str):
        """
        Loads the rest_test data from the specified file and initializes an empty dictionary for user histories.

        Args:
            rest_history_file_path (str): Path to the rest_test.npz file.
        """
        self.rest_history = load_npz(rest_history_file_path)
        self.user_histories = {}  # Dictionary to store user histories

    def get_user_actual(self, userId: int):
        """
        Extracts the app_ids (column indices) from the rest_test history for the given user.

        Args:
            userId (int): The ID of the user whose history is requested.

        Returns:
            np.ndarray: A numpy array containing the app_ids the user interacted with (column indices).
        """
        # Extract row and column data from COO matrix
        user_row = self.rest_history.row
        user_col = self.rest_history.col

        # Filter entries based on the user ID
        user_entries = user_col[user_row == userId]

        return user_entries

    def build_user_histories(self):
        """
        Iterates through the rest_test data and builds a dictionary where the key is the user_id and the value is a list of app_ids (column indices) they interacted with.
        """
        user_row = self.rest_history.row
        user_col = self.rest_history.col

        # Loop through each entry in rest_test data
        for row, col in zip(user_row, user_col):
            if row not in self.user_histories:
                self.user_histories[row] = []
            self.user_histories[row].append(col)


def user_test_ids(test_matrix_path):
    '''
    parameters:
        test_matrix_path (str): path to the test matrix file
    returns:
        list of user_id in test matrix
    '''
    pass


def funk_svd_testing():
    import importlib.util
    from pathlib import Path

    module_path = Path("funk-svd/model.py")
    spec = importlib.util.spec_from_file_location("model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Testing = module.Testing

    test_instance = Testing('./', 'train_and_test.npz')

    test_user_ids = user_test_ids('test_matrix.npz')
    history_getter = HistoryGetter('rest_test.npz')

    #TODO ---------
    user_amount = 1 
    recommendation_size = 1 
    all_items = 1
    #-------------

    for user_id in test_user_ids:
        predicted = test_instance.ask_for_recommendation(user_id, 10)
        actual = history_getter.get_user_actual(user_id)
        test_results = test_model(predicted, actual, k=10, user_amount=user_amount, recommendation_size=recommendation_size, all_items=all_items)
        print(test_results)


# funk_svd_testing()



abc = HistoryGetter('rest_test.npz')
abc.build_user_histories()
print(abc.get_user_actual(5))