from typing import List
import importlib.util
import math
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
import tqdm
import HistoryGetter
from metrics_helpers import get_user_test_ids, load_all_items_id

#1.
def recall(predicted: List[List], actual: List[List], k: int) -> float:
    '''
    parameters:
        predicted: List of lists containing the top k recommendations for each user in test set 
        actual: List of lists containing the actual items interacted with by each user in test set
        k: int, number of recommendations to consider
    returns:
        float: recall score for the model calculated as the number of relevant items recommended to the user divided by the total number of relevant items - how many of actual[i] are in predicted[i] divided by len(actual[k])
    '''
    total_relevant = 0
    relevant_recommended = 0

    for pred, act in zip(predicted, actual):
        if len(act) == 0:  # Avoid division by zero when actual is empty
            continue
        act_set = set(tuple(act)) 
        pred_k = pred[:k] 
        relevant_recommended += len([item for item in pred_k if item in act_set])
        total_relevant += len(act)

    if total_relevant == 0:
        return 0.0

    return relevant_recommended / total_relevant

#2.

def hitrate(predicted: List[List], actual: List[List], k) -> float:
    '''
    parameters:
        predicted - recommendations for each user
        actual - actual items for each user (relevant items)
        k - number of recommendations to consider
    returns:
        how many users how at least one relevant item in top k recommendations
    '''
    assert(len(predicted[0]) >= k)
    def helper(predicted, actual, k):
        predicted_set = set(tuple(item) for item in predicted[:k])
        actual_set = set(tuple(item) for item in actual)
        if len(predicted_set & actual_set) > 0:
            return 1 
        return 0
    total = np.sum([helper(predicted, actual, k) for i in range(len(predicted))])
    return total / len(predicted)

def MRR(predicted: List[List], actual: List[List], k: int) -> float:
    '''
    parameters:
        predicted - recommendations for each user
        actual - actual items for each user (relevant items)
        k - number of recommendations to consider
    returns:
        float: MRR score for the model.
    '''
    if len(predicted) == 0 or len(actual) == 0 or len(predicted) != len(actual):
        raise ValueError("Predicted and actual lists must have the same length and cannot be empty.")

    reciprocal_ranks = []

    for pred, act in zip(predicted, actual):
        act_set = {tuple([item]) if np.isscalar(item) else tuple(item) for item in act}
        for rank, item in enumerate(pred[:k], start=1):
            item_tuple = tuple([item]) if np.isscalar(item) else tuple(item)
            if item_tuple in act_set:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0) 

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def NDCG(predicted: List[List], actual: List[List], k: int) -> float:
    '''
    parameters:
        predicted - recommendations for each user
        actual - actual items for each user (relevant items)
        k - number of recommendations to consider
    returns:
        NDCG score for the model calculated as the sum of the DCG scores for each user divided by the total number of users
    '''
    def dcg(pred, act, k):
        dcg_score = 0
        act_set = {tuple(item) if isinstance(item, list) else item for item in act}
        for i in range(min(k, len(pred))):
            item = tuple(pred[i]) if isinstance(pred[i], list) else pred[i]
            if item in act_set:
                dcg_score += 1 / (np.log2(i + 2))  # i+2 because log starts at position 1 (index 0)
        return dcg_score

    def idcg(act, k):
        idcg_score = 0
        for i in range(min(k, len(act))):
            idcg_score += 1 / (np.log2(i + 2))
        return idcg_score

    total = np.sum([dcg(pred, act, k) / idcg(act, k) if idcg(act, k) > 0 else 0 
                    for pred, act in zip(predicted, actual)])
    return total / len(predicted)

#3.

def catalog_coverage(predicted: List[List[int]], all_items: List[int], k: int) -> float:
    """
    Calculates the catalog coverage for a recommendation model.

    Parameters:
        predicted (List[List[int]]): List of lists containing the top k recommendations for each user.
        all_items (List[int]): List of all possible items in the catalog.
        k (int): Number of recommendations to consider.

    Returns:
        float: Catalog coverage, calculated as the proportion of unique recommended items in the catalog.
    """
    if not predicted or len(predicted) == 0:
        raise ValueError("Predicted list cannot be empty.")
    if all_items is None or len(all_items) == 0:
        raise ValueError("all_items cannot be empty.")

    recommended_items = set()

    for user_recommendations in predicted:
        recommended_items.update(user_recommendations[:k])  # Consider only the top-k recommendations

    coverage = len(recommended_items) / len(all_items)
    return coverage

def novelty(predicted: List[List[int]], actual: List[List[int]]) -> float:
    """
    Calculates the novelty score for a recommendation model.

    Parameters:
        predicted (List[List[int]]): List of lists containing the recommendations for each user.
        actual (List[List[int]]): List of lists containing the actual items interacted with by each user.

    Returns:
        float: Novelty score for the model, based on the average inverse popularity of recommended items.
    """
    if len(predicted) == 0 or len(actual) == 0 or len(predicted) != len(actual):
        raise ValueError("Predicted and actual lists must have the same length and cannot be empty.")

    # Compute item popularity (how often each item appears in actual interactions)
    item_popularity = {}
    for user_actual in actual:
        for item in user_actual:
            item_popularity[item] = item_popularity.get(item, 0) + 1

    total_users = len(actual)
    total_novelty = 0
    total_recommendations = 0

    for user_recommendations in predicted:
        for item in user_recommendations:
            if item in item_popularity:
                # Popularity is normalized by total users to get the probability of interaction
                popularity = item_popularity[item] / total_users
                total_novelty += -math.log2(popularity)  # Higher inverse popularity is more novel
                total_recommendations += 1

    if total_recommendations == 0:
        return 0.0  # Avoid division by zero

    return total_novelty / total_recommendations


#---------

def test_model(predicted: List[List], actual: List[List], k: int, all_items: List):
    recall_score = recall(predicted, actual, k)
    hitrate_score = hitrate(predicted, actual, k)
    MRR_score = MRR(predicted, actual, k)
    NDCG_score = NDCG(predicted, actual, k)
    catalog_coverage_score = catalog_coverage(predicted, all_items, k)
    novelty_score = novelty(predicted, actual)
    return {"recall": recall_score, "hitrate": hitrate_score, "MRR": MRR_score, "NDCG": NDCG_score, "catalog_coverage": catalog_coverage_score, "novelty": novelty_score}

def model_testing(model, k=10):
    all_items = load_all_items_id('games.csv')
    historyGetter = HistoryGetter.HistoryGetter('rest_test.npz')
    predictedList = []
    actualList = []
    test_user_ids = get_user_test_ids('test_matrix.npz')

    amount_to_recommend = 2*k 
    if amount_to_recommend > 10000:
        amount_to_recommend = k+1
    exit_limit = 200
    exit_count = 0
    for user_id in tqdm.tqdm(test_user_ids, total=len(test_user_ids), desc="Processing Users"):
        exit_count += 1
        predicted = model.ask_for_recommendation(user_id, k)
        new_predicted = []

        for item in predicted:  
            new_predicted.append(item['app_id']) 

        predicted = new_predicted
        actual = historyGetter.get_user_actual(user_id)

        predictedList.append(predicted)
        actualList.append(actual)
        if exit_count == exit_limit:
            break
    test_results = test_model(predictedList, actualList, k, all_items=all_items)
    print(test_results)


def funksvd_testing():
    module_path = Path("funk-svd/model2.py")
    spec = importlib.util.spec_from_file_location("model2", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Testing = module.Testing
    test_instance = Testing('./', 'train_and_test.npz')

    model_testing(test_instance, k=1000)

funksvd_testing()

# class randomModel():
#     def __init__(self):
#         self.games = pd.read_csv('games.csv')
#         self.unique_appids = self.games['app_id'].unique()
        
#     def ask_for_recommendation(self, user_id, k):
#         return np.random.choice(self.unique_appids, k, replace=False).tolist()
    
# model_testing(randomModel(), k=1000)


