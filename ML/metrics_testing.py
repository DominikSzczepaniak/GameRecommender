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
        if not act:  # Avoid division by zero when actual is empty
            continue
        act_set = set(act) 
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
    assert(len(predicted) >= k)
    def helper(predicted, actual, k):
        if len(set(predicted[:k]) & set(actual)) > 0:
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
    if not predicted or not actual or len(predicted) != len(actual):
        raise ValueError("Predicted and actual lists must have the same length and cannot be empty.")

    reciprocal_ranks = []

    for pred, act in zip(predicted, actual):
        act_set = set(act) 
        for rank, item in enumerate(pred[:k], start=1):
            if item in act_set:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)  # No relevant items in the top-k recommendations

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
    def dcg(predicted, actual, k):
        dcg = 0
        for i in range(k):
            if predicted[i] in actual:
                dcg += 1 / (np.log2(i + 2))
        return dcg

    def idcg(actual, k):
        idcg = 0
        for i in range(k):
            idcg += 1 / (np.log2(i + 2))
        return idcg

    total = np.sum([dcg(predicted, actual, k) / idcg(actual, k) for i in range(len(predicted))])
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
    if not predicted or not all_items:
        raise ValueError("Predicted and all_items lists cannot be empty.")

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
    if not predicted or not actual or len(predicted) != len(actual):
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
    historyGetter = HistoryGetter('rest_test.npz')
    predictedList = []
    actualList = []
    test_user_ids = get_user_test_ids('test_matrix.npz')

    for user_id in tqdm.tqdm(test_user_ids, total=len(test_user_ids), desc="Processing Users"):
        predicted = model.ask_for_recommendation(user_id, 10)
        new_predicted = []

        for item in predicted:  
            new_predicted.append(item['app_id']) 

        predicted = new_predicted
        actual = historyGetter.get_user_actual(user_id)

        predictedList.append(predicted)
        actualList.append(actual)

    test_results = test_model(predictedList, actualList, k, all_items=all_items)
    print(test_results)


def funksvd_testing():
    module_path = Path("funk-svd/model2.py")
    spec = importlib.util.spec_from_file_location("model2", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Testing = module.Testing
    test_instance = Testing('./', 'train_and_test.npz')

    model_testing(test_instance, k=10)

funksvd_testing()



