from recmetrics import novelty, prediction_coverage, catalog_coverage, _ark, _apk, mark, mapk, personalization

#diversity = catalog_coverage 
#novelty = novelty
#coverage popularity = prediction coverage

def NDCG(predicted: List[List], actual: List[List], k: int):
    pass

def test_model(predicted: List[List], actual: List[List], k: int, user_amount: int, recommendation_size: int, all_items: List):
    novelty_score = novelty(predicted, actual)
    prediction_coverage_score = prediction_coverage(predicted, actual)
    catalog_coverage_score = catalog_coverage(predicted, all_items, k)
    personalization_score = personalization(predicted)
    NDCG_score = NDCG(predicted, actual, k) #to change if needed 
    ark_score = _ark(predicted, actual, k)
    MAP_score = mapk(predicted, actual, k)
    return {"novelty": novelty_score, "prediction_coverage": prediction_coverage_score, "catalog_coverage": catalog_coverage_score, "personalization": personalization_score, "NDCG": NDCG_score, "ark": ark_score, "MAP": MAP_score}