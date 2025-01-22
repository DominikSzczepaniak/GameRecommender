from typing import List
import importlib.util
import math
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
import tqdm
from history_getter import HistoryGetter

# -----------------------=[ METRICS ]=-----------------------

'''
  Recall@k

  Measures: How many of the actual items for a user are present in the top-k recommendations.
  Focus: Recovers relevant items from the user's history.
'''

def recall(predicted: List[List], actual: List[List], k: int) -> float:
  total_relevant = 0
  relevant_recommended = 0

  for pred, act in zip(predicted, actual):

    act_set = set(tuple(act)) 
    pred_k = pred[:k] 

    relevant_recommended += len([item for item in pred_k if item in act_set])
    total_relevant += len(act)

  if total_relevant == 0:
    return 0.0

  return relevant_recommended / total_relevant

'''
    Hit Rate@k

    Measures: The proportion of users for whom at least one of their actual items is present in the top-k recommendations.
    Focus: Simple check for any overlap between actual and recommended items.
'''

def hitrate(predicted: List[List], actual: List[List], k: int) -> float:
  hits = 0

  for pred, act in zip(predicted, actual):
    # Check if there's any overlap between top-k predicted and actual items
    if any(item in act for item in pred[:k]):
      hits += 1

  return hits / len(predicted) if predicted else 0.0

'''
  MRR@k

  Measures: The average reciprocal rank of the first relevant item in the recommendation list.
  Focus: Rewards systems that rank relevant items higher.
'''

def MRR(predicted: List[List], actual: List[List], k: int) -> float:
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

'''
  NDCG@k

  Measures: The normalized discounted cumulative gain, which considers both the relevance of 
  recommended items and their position in the list.
  
  Focus: Prioritizes relevant items at the top of the list and penalizes irrelevant items.
'''

def NDCG(predicted: List[List], actual: List[List], k: int) -> float:
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

'''
  Catalog Coverage@k

  Measures: The proportion of items in the catalog that are recommended to at least one user within the top-k recommendations.
  Focus: Assesses the diversity of recommendations across the entire item catalog.
'''

def catalog_coverage(predicted: List[List[int]], all_items: List[int], k: int) -> float:
  recommended_items = set()

  for user_recommendations in predicted:
    recommended_items.update(user_recommendations[:k])  # Consider only the top-k recommendations

  coverage = len(recommended_items) / len(all_items)
  return coverage

'''
  Novelty

  Measures: The average inverse popularity of recommended items.
  Focus: Rewards systems that recommend less frequently interacted items, promoting exploration.
'''

def novelty(predicted: List[List[int]], actual: List[List[int]]) -> float:
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

# -----------------------=[ MODEL EVALUATION ]=-----------------------

def test_model(predicted: List[List], actual: List[List], k: int, all_items: List):
  if len(predicted) != len(actual):
      raise ValueError("Predicted and actual lists must have the same length.")

  recall_score = recall(predicted, actual, k)
  hitrate_score = hitrate(predicted, actual, k)

  MRR_score = MRR(predicted, actual, k)
  NDCG_score = NDCG(predicted, actual, k)

  catalog_coverage_score = catalog_coverage(predicted, all_items, k)
  novelty_score = novelty(predicted, actual)

  return {"recall": recall_score, "hitrate": hitrate_score, 
          "MRR": MRR_score, "NDCG": NDCG_score, 
          "catalog_coverage": catalog_coverage_score, "novelty": novelty_score}


def model_testing(model, path, k=10):
  historyGetter = HistoryGetter(path)
  all_items = historyGetter.load_all_items_id()
  predictedList = []
  actualList = []
  test_user_ids = historyGetter.get_user_test_ids()

  counter =0 

  for user_id in tqdm.tqdm(test_user_ids, total=len(test_user_ids), desc="Processing Users"):
    predicted = model.ask_for_recommendation(user_id, k + 1)

    actual = historyGetter.get_user_actual(user_id)

    predictedList.append(predicted)
    actualList.append(actual)

    counter += 1
    if counter > 20:
      break

  test_results = test_model(predictedList, actualList, k, all_items=all_items)

  print(test_results)

  return test_results


def funksvd_testing(k):
  from funksvd.model2 import Testing

  model_testing(Testing('./', 'train_and_test.npz'), k=10)


def lightfm_testing(k):
  from lightFMscaNN.model import LightFMscaNN

  model_testing(LightFMscaNN(), "lightFMscaNN", k)


def baseline_testing(k):
  class randomModel():
    def __init__(self):
      self.games = pd.read_csv('games.csv')
      self.unique_appids = self.games['app_id'].unique()
      
    def ask_for_recommendation(self, user_id, k):
      return np.random.choice(self.unique_appids, k, replace=False).tolist()

  res = []

  # Do it 1000x times
  for i in range(1000):
    x = model_testing(randomModel(), k)
    res.append((x['recall'], x['MRR'], x['hitrate']))

  recalls, MRRs = zip(*res)

  mean_recall = np.mean(recalls)
  mean_mrr = np.mean(MRRs)

  print(f"Mean Recall: {mean_recall}, Mean MRR: {mean_mrr}")


# funksvd_testing(20)
lightfm_testing(20)


'''
OBSŁUGA:
Podajesz w konstruktorze nazwę folderu z twoim modelem.

W folderze tym musi znajdować się folder data z:
  users.csv
  games.csv
  rest_test.npz
  test_matrix.npz

Kod odpalany jest z folderu ML, więc w swoim pliku model.py, jeśli otwierasz jakieś pliki,
to każdy path zaczyna się z ./<nazwa_twojego_folderu>/data/...
'''
