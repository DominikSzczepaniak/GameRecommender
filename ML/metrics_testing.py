from recmetrics import novelty, prediction_coverage, catalog_coverage, _ark, _apk, mark, mapk, personalization

#diversity = catalog_coverage 
#novelty = novelty
#coverage popularity = prediction coverage
def hitrate(predicted: List[List], actual: List[List], k):
    def hitrate_for_user(predicted, actual, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        for item in predicted:
            if item in actual:
                return 1
        return 0
    
    hits = 0 
    for i in range(len(predicted)):
        hits += hitrate_for_user(predicted[i], actual[i], k)
    return hits / len(predicted)