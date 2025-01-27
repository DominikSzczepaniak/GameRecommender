"""
This module calculates recommendation system evaluation metrics including NDCG, Hit Rate, Recall, and Precision.

You need ./data/rest_test.npz and ./data/train_and_test.npz
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from tqdm import tqdm
from typing import List


def mrr_k(y_true, y_pred):
    """
    Calculates Mean Reciprocal Rank (MRR@k)

    Args:
        y_true: List of arrays of true relevant item IDs
        y_pred: List of arrays of recommended item IDs

    Returns:
        Mean Reciprocal Rank score (0-1)
    """
    reciprocal_ranks = []
    for true_items, recommended in zip(y_true, y_pred):
        # Find first occurrence of a relevant item in recommendations
        ranks = np.where(np.isin(recommended, true_items))[0]
        if ranks.size > 0:
            reciprocal_ranks.append(
                1 / (ranks[0] + 1))  # +1 because ranks start at 1
        else:
            reciprocal_ranks.append(0)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def catalogue_coverage(y_pred, total_items):
    """
    Calculates the percentage of unique items recommended across all users.

    Args:
        y_pred: List of arrays containing recommended item IDs for each user
        total_items: Total number of unique items in the catalogue

    Returns:
        Coverage percentage (0-1) of unique items recommended
    """
    all_recommended = np.concatenate(y_pred)
    unique_items = np.unique(all_recommended)

    return len(unique_items) / total_items


def ndcg_k(test_users, y_pred, test_interactions, train_interactions):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG@k)

    Args:
        test_users: List of test user IDs
        y_pred: List of arrays of recommended item IDs for each user
        test_interactions: Sparse matrix of test user-item interactions
        train_interactions: Sparse matrix of training user-item interactions

    Returns:
        Average NDCG@k across all test users
    """
    ndcg_scores = []

    for user_id, recommended_items in zip(tqdm(test_users), y_pred):
        true_positives = test_interactions[user_id].indices
        train_positives = train_interactions[user_id].indices
        true_positives = np.setdiff1d(true_positives, train_positives)

        relevance = np.isin(recommended_items, true_positives).astype(float)

        dcg = 0.0
        for pos, rel in enumerate(relevance):
            dcg += rel / np.log2(pos + 2)

        ideal_relevance = np.zeros_like(relevance)
        ideal_relevance[:min(len(recommended_items), len(true_positives))] = 1.0
        idcg = 0.0
        for pos, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(pos + 2)

        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def hitrate_k(test_users, y_pred, test_interactions, train_interactions):
    """
    Calculates Hit Rate@k - percentage of users with at least one relevant recommendation

    Args:
        test_users: List of test user IDs
        y_pred: List of arrays of recommended item IDs for each user
        test_interactions: Sparse matrix of test interactions
        train_interactions: Sparse matrix of training interactions

    Returns:
        Hit rate percentage (0-1)
    """
    hits = 0

    for user_id, recommended_items in zip(tqdm(test_users), y_pred):
        true_positives = test_interactions[user_id].indices
        train_positives = train_interactions[user_id].indices
        true_positives = np.setdiff1d(true_positives, train_positives)

        if len(np.intersect1d(recommended_items, true_positives)) > 0:
            hits += 1

    return hits / len(test_users) if len(test_users) > 0 else 0.0


def recommender_recall(predicted: List[list], actual: List[list]) -> int:
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """
    def calc_recall(predicted, actual):
        reca = [value for value in predicted if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

    recall = np.mean(list(map(calc_recall, predicted, actual)))
    return recall

def recommender_precision(predicted: List[list], actual: List[list]) -> int:
    """
    Computes the precision of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        precision: int
    """
    def calc_precision(predicted, actual):
        prec = [value for value in predicted if value in actual]
        prec = np.round(float(len(prec)) / float(len(predicted)), 4)
        return prec

    precision = np.mean(list(map(calc_precision, predicted, actual)))
    return precision

def test_metrics(model, k):
    """
    Main evaluation function that calculates multiple metrics

    Args:
        model: Trained recommendation model
        k: Number of recommendations to evaluate

    Returns:
        Dictionary of metrics including Recall@k, HitRate@k, Precision@k, NDCG@k, Coverage, and MRR@k
    """

    test_interactions = load_npz("./data/rest_test.npz").tocsr()
    train_interactions = load_npz("./data/train_and_test.npz").tocsr()

    total_items = train_interactions.shape[1]

    test_users = np.unique(test_interactions.nonzero()[0])
    y_true = []
    y_pred = []

    # random_subset = np.random.choice(test_users, size=300, replace=False)
    # test_users = random_subset

    for user_id in tqdm(test_users):
        true_positives = test_interactions[user_id].indices
        train_positives = train_interactions[user_id].indices
        y_true.append(np.setdiff1d(true_positives, train_positives))
        y_pred.append(model.recommend(user_id, k))

    recall = recommender_recall(y_pred, y_true)
    hitrate = hitrate_k(test_users, y_pred, test_interactions, train_interactions)
    precision = recommender_precision(y_pred, y_true)
    ndcg = ndcg_k(test_users, y_pred, test_interactions, train_interactions)
    coverage = catalogue_coverage(y_pred, total_items)
    mrr = mrr_k(y_true, y_pred)

    return {
        "recall": recall,
        "hitrate": hitrate,
        "precision": precision,
        "ndcg": ndcg,
        "coverage": coverage,
        "mrr": mrr,
    }
    
from model import SVDppCustom

print(test_metrics(SVDppCustom('./data/train_and_test.npz'), 20))