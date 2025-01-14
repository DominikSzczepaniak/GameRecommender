from recmetrics import novelty, prediction_coverage, catalog_coverage, _ark, _apk, mark, mapk, personalization
from typing import List
import numpy as np
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