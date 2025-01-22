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
  count = 0
  for user_id in tqdm.tqdm(test_user_ids, total=len(test_user_ids), desc="Processing Users"):
    predicted = model.ask_for_recommendation(user_id, k + 1)

    actual = historyGetter.get_user_actual(user_id)

    predictedList.append(predicted)
    actualList.append(actual)
    count += 1
    if count > 20:
      break

  test_results = test_model(predictedList, actualList, k, all_items=all_items)

  print(test_results)

  return test_results


def funksvd_testing(k):
  # DO POPRAWY DOMINO
  module_path = Path("funk-svd/model2.py")
  spec = importlib.util.spec_from_file_location("model2", module_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  Testing = module.Testing
  test_instance = Testing('./', 'train_and_test.npz')

  model_testing(test_instance, k=10)


def lightfm_testing(k):
  from lightFMscaNN.model import LightFMscaNN

  model_testing(LightFMscaNN(k + 1), "lightFMscaNN", k)

def widendeep_testing(k):
  from widendeep.model import Widendeep

  model_testing(Widendeep(), "widendeep", k)
  
def cleora_testing(k):
  from cleora.model import Cleora 
  
  model_testing(Cleora(), "cleora", k)


def baseline_testing(k):
  class randomModel():
    def __init__(self):
      self.games = pd.read_csv('./lightFMscaNN/data/games.csv')
      self.unique_appids = self.games['app_id'].unique()
      
    def ask_for_recommendation(self, user_id, k):
      return np.random.choice(self.unique_appids, k, replace=False).tolist()

  res = []
  randModel = randomModel()

  # Do it 100x times
  for i in range(100):
    x = model_testing(randModel,'./lightFMscaNN',  k)
    res.append((x['recall'], x['hitrate'], x['MRR'], x['NDCG'], x['catalog_coverage'], x['novelty']))

  recalls, hitrates, MRRs, NDCGs, ccs, novelties = zip(*res)

  mean_recall = np.mean(recalls)
  mean_hitrate = np.mean(hitrates)
  mean_mrr = np.mean(MRRs)
  mean_ndcg = np.mean(NDCGs)
  mean_cc = np.mean(ccs)
  mean_novelty = np.mean(novelties)

  print(f"recall: {mean_recall}\nhitrate: {mean_hitrate}\nMRR: {mean_mrr}\nNDCG: {mean_ndcg}\ncatalog_coverage: {mean_cc}\nnovelty: {mean_novelty}")


# funksvd_testing(20)
# lightfm_testing(1000)
cleora_testing(20)
# baseline_testing(5)

# --------------=[ RANDOM MODEL (BASELINE) ]=-------------------

# MEAN RESULT k = 1000
# recall: 0.0195531901726592
# hitrate: 0.133535
# MRR: 0.0011883266309552643
# NDCG: 0.004356446537054758
# catalog_coverage: 1.0
# novelty: 9.985290890104652

# MEAN RESULT k = 100
# recall: 0.001962373909276564
# hitrate: 0.015565000000000002
# MRR: 0.0007569879781817824
# NDCG: 0.0007428989669030387
# catalog_coverage: 0.9805128557949362
# novelty: 9.983184253399171

# MEAN RESULT k = 50
# recall: 0.0009629308744352992
# hitrate: 0.007715
# MRR: 0.0006583595102567189
# NDCG: 0.00045173036030064925
# catalog_coverage: 0.8599396524610787
# novelty: 9.98752393814248

# MEAN RESULT k = 20
# recall: 0.00039915836376013355
# hitrate: 0.003215
# MRR: 0.0005434982047536228
# NDCG: 0.00026699828372699476
# catalog_coverage: 0.544651871363422
# novelty: 9.98512749067507

# MEAN RESULT k = 5
# recall: 9.901602821956804e-05
# hitrate: 0.0008000000000000001
# MRR: 0.0003858333333333333
# NDCG: 0.0001708630981299354
# catalog_coverage: 0.17859175971064634
# novelty: 9.986698678624087

# ----------------=[ FUNK SVD ]=-------------------

# RESULT FOR K = 1000
# recall: 0.1871906129682821
# hitrate: 0.683
# MRR: 0.00481212914512411
# NDCG: 0.051431921625534525
# catalog_coverage: 0.586353986475861
# novelty: 9.501393118104643

# ----------------=[ LIGHTFM & SCANN ]=-------------------

# RESULT FOR K = 1000
# recall: 0.3153660498793242
# hitrate: 0.7335 
# MRR: 0.04248862490598973
# NDCG: 0.09736259528409646
# catalog_coverage: 0.3226922472086806
# novelty: 8.893390041598554

# RESULT FOR K = 100
# recall: 0.07958413268147782
# hitrate: 0.386 
# MRR: 0.04248862490598973
# NDCG: 0.04106378410285326
# catalog_coverage: 0.10245321591445196
# novelty: 7.737388446837643

# RESULT FOR K = 50
# recall: 0.04746580852775543
# hitrate: 0.2745
# MRR: 0.039523982299902206
# NDCG: 0.030717454205468276
# catalog_coverage: 0.07100172983173456
# novelty: 7.474290710517738

# RESULT FOR K = 20
# recall: 0.02258803143758896
# hitrate: 0.153
# MRR: 0.035851715720639936
# NDCG: 0.01996750158017846
# catalog_coverage: 0.04306887875452115
# novelty: 7.243296084036996

# RESULT FOR K = 5
# recall: 0.007240547063555913
# hitrate: 0.054
# MRR: 0.026791666666666686
# NDCG: 0.014019246466019422
# catalog_coverage: 0.020561409026576504
# novelty: 6.980510243720895

# ----------------=[ Wide & Deep Neural Network ]=-------------------

# RESULT FOR K = 1000
# recall: 0.6317912363258571
# hitrate: 0.9565
# MRR: 0.0950702559634543
# NDCG: 0.20419445006575007
# catalog_coverage: 0.033554804214499134
# novelty: 8.136490434583198

# RESULT FOR K = 20
# recall: 0.054574344557843914,
# hitrate: 0.315
# MRR: 0.08526039451819249
# NDCG: 0.0521284034010134
# catalog_coverage: 0.0013170309797137915
# novelty: 5.558717925101877


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
