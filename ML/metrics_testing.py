# from recmetrics import novelty, prediction_coverage, catalog_coverage, personalization
from typing import List
import numpy as np
from scipy.sparse import load_npz, coo_matrix
import pandas as pd
import tqdm
import random
#diversity = catalog_coverage 
#novelty = novelty
#coverage popularity = prediction coverage

import random
from itertools import product
from math import sqrt
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import warnings


def novelty(predicted: List[list], pop: dict, u: int, n: int) -> (float, list):
    """
    Computes the novelty for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    n: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    mean_self_information:
        The novelty of the recommendations in recommended top-N list level
    ----------    
    Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    mean_self_information = []
    k = 0
    for sublist in predicted:
        self_information = 0
        k += 1
        for i in sublist:
            self_information += np.sum(-np.log2(pop[i]/u))
        mean_self_information.append(self_information/n)
    novelty = sum(mean_self_information)/k
    return novelty, mean_self_information

def prediction_coverage(predicted: List[list], catalog: list, unseen_warning: bool=False) -> float:
    """
    Computes the prediction coverage for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    unseen_warn: bool
        when prediction gives any item unseen in catalog: 
            (1) ignore the unseen item and warn
            (2) or raise an exception.
    Returns
    ----------
    prediction_coverage:
        The prediction coverage of the recommendations as a percent
        rounded to 2 decimal places
    ----------    
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    
    unique_items_catalog = set(catalog)
    if len(catalog)!=len(unique_items_catalog):
        raise AssertionError("Duplicated items in catalog")

    predicted_flattened = [p for sublist in predicted for p in sublist]
    unique_items_pred = set(predicted_flattened)
    
    if not unique_items_pred.issubset(unique_items_catalog):
        if unseen_warning:
            warnings.warn("There are items in predictions but unseen in catalog. "
                "They are ignored from prediction_coverage calculation")
            unique_items_pred = unique_items_pred.intersection(unique_items_catalog)
        else:
            raise AssertionError("There are items in predictions but unseen in catalog.")
    
    num_unique_predictions = len(unique_items_pred)
    prediction_coverage = round(num_unique_predictions/(len(catalog)* 1.0)* 100, 2)
    return prediction_coverage

def catalog_coverage(predicted: List[list], catalog: list, k: int) -> float:
    """
    Computes the catalog coverage for k lists of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    k: integer
        The number of observed recommendation lists
        which randomly choosed in our offline setup
    Returns
    ----------
    catalog_coverage:
        The catalog coverage of the recommendations as a percent
        rounded to 2 decimal places
    ----------    
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    sampling = random.choices(predicted, k=k)
    predicted_flattened = [p for sublist in sampling for p in sublist]
    L_predictions = len(set(predicted_flattened))
    catalog_coverage = round(L_predictions/(len(catalog)*1.0)*100,2)
    return catalog_coverage

def _ark(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average recall at k.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def _apk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average precision at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : float
        The average precision at k.
    """
    if not predicted or not actual:
        return 0.0
    
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    true_positives = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            max_ix = min(i + 1, len(predicted))
            score += _precision(predicted[:max_ix], actual)
            true_positives += 1
    
    if score == 0.0:
        return 0.0
    
    return score / true_positives

def mark(actual: List[list], predicted: List[list], k=10) -> float:
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average recall at k (mar@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")

    return np.mean([_ark(a,p,k) for a,p in zip(actual, predicted)])

def mapk(actual: List[list], predicted: List[list], k: int=10) -> float:
    """
    Computes the mean average precision at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: float
            The mean average precision at k (map@k)
    """
    if len(actual) != len(predicted):
        raise AssertionError("Length mismatched")
    
    return np.mean([_apk(a,p,k) for a,p in zip(actual, predicted)])

def personalization(predicted: List[list]) -> float:
    """
    Personalization measures recommendation similarity across users.
    A high score indicates good personalization (user's lists of recommendations are different).
    A low score indicates poor personalization (user's lists of recommendations are very similar).
    A model is "personalizing" well if the set of recommendations for each user is different.
    Parameters:
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        The personalization score for all recommendations.
    """

    def make_rec_matrix(predicted: List[list]) -> sp.csr_matrix:
        df = pd.DataFrame(data=predicted).reset_index().melt(
            id_vars='index', value_name='item',
        )
        df = df[['index', 'item']].pivot(index='index', columns='item', values='item')
        df = pd.notna(df)*1
        rec_matrix = sp.csr_matrix(df.values)
        return rec_matrix

    #create matrix for recommendations
    predicted = np.array(predicted)
    rec_matrix_sparse = make_rec_matrix(predicted)

    #calculate similarity for every user's recommendation list
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

    #calculate average similarity
    dim = similarity.shape[0]
    personalization = (similarity.sum() - dim) / (dim * (dim - 1))
    return 1-personalization

def _single_list_similarity(predicted: list, feature_df: pd.DataFrame, u: int) -> float:
    """
    Computes the intra-list similarity for a single list of recommendations.
    Parameters
    ----------
    predicted : a list
        Ordered predictions
        Example: ['X', 'Y', 'Z']
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
    ils_single_user: float
        The intra-list similarity for a single list of recommendations.
    """
    # exception predicted list empty
    if not(predicted):
        raise Exception('Predicted list is empty, index: {0}'.format(u))

    #get features for all recommended items
    recs_content = feature_df.loc[predicted]
    recs_content = recs_content.dropna()
    recs_content = sp.csr_matrix(recs_content.values)

    #calculate similarity scores for all items in list
    similarity = cosine_similarity(X=recs_content, dense_output=False)

    #get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)

    #calculate average similarity score of all recommended items in list
    ils_single_user = np.mean(similarity[upper_right])
    return ils_single_user

def intra_list_similarity(predicted: List[list], feature_df: pd.DataFrame) -> float:
    """
    Computes the average intra-list similarity of all recommendations.
    This metric can be used to measure diversity of the list of recommended items.
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        Example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
        The average intra-list similarity for recommendations.
    """
    feature_df = feature_df.fillna(0)
    Users = range(len(predicted))
    ils = [_single_list_similarity(predicted[u], feature_df, u) for u in Users]
    return np.mean(ils)

def mse(y: list, yhat: np.array) -> float:
    """
    Computes the mean square error (MSE)
    Parameters
    ----------
    yhat : Series or array. Reconstructed (predicted) ratings or interaction values.
    y: original true ratings or interaction values.
    Returns:
    -------
        The mean square error (MSE)
    """
    mse = mean_squared_error(y, yhat)
    return mse

def rmse(y: list, yhat: np.array) -> float:
    """
    Computes the root mean square error (RMSE)
    Parameters
    ----------
    yhat : Series or array. Reconstructed (predicted) ratings or values
    y: original true ratings or values.
    Returns:
    -------
        The root mean square error (RMSE)
    """
    rmse = sqrt(mean_squared_error(y, yhat))
    return rmse

def make_confusion_matrix(y: list, yhat: list) -> None:
    """
    Calculates and plots a confusion matrix
    Parameters
    ----------
    y : list or array of actual interaction values such as ratings
    yhat: list or array of actual predicted interaction values
    Returns:
    -------
        A confusion matrix plot
    """
    cm = confusion_matrix(y, yhat, labels=[1,0])
    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],4)*100

    fmt = ".2f"
    _ = cm.max() / 2. # TODO: Unused argument
    descriptions = np.array([["True Positive", "False Negative"], ["False Positive", "True Negatives"]])
    colors = np.array([["green", "red"], ["red", "green"]])
    plt.imshow([[0,0],[0,0]], interpolation='nearest', cmap=plt.cm.Greys)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt)+'%\n' + descriptions[i, j],
                     horizontalalignment="center",
                     color=colors[i,j])
    plt.axhline(y=0.5, xmin=0, xmax=1, color="black", linewidth=0.75)
    plt.axvline(x=0.5, ymin=0, ymax=1, color="black", linewidth=0.75)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.xticks([0,1], [1,0], rotation=45)
    plt.yticks([0,1], [1,0])
    plt.show()

def _precision(predicted, actual):
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec

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
 
    precision = np.mean(list(map(lambda x, y: np.round(_precision(x,y), 4), predicted, actual)))
    return precision


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
    

def test_model(predicted: List[List], actual: List[List], k: int, all_items: List):
    # novelty_score = novelty(predicted, actual)
    prediction_coverage_score = prediction_coverage(predicted, all_items)
    catalog_coverage_score = catalog_coverage(predicted, all_items, k)
    personalization_score = personalization(predicted)
    NDCG_score = NDCG(predicted, actual, k) #to change if needed 
    # ark_score = _ark(predicted, actual, k)
    # MAP_score = mapk(predicted, actual, k)
    return {"prediction_coverage": prediction_coverage_score, "catalog_coverage": catalog_coverage_score, "personalization": personalization_score, "NDCG": NDCG_score}

class HistoryGetter:
    def __init__(self, rest_history_file_path: str):
        """
        Loads the rest_test data from the specified file and initializes an empty dictionary for user histories.

        Args:
            rest_history_file_path (str): Path to the rest_test.npz file.
        """
        self.rest_history = load_npz(rest_history_file_path)
        self.user_histories = {}  # Dictionary to store user histories
        self.load_mappings()

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
        for i in range(len(user_entries)):
            user_entries[i] = self.reverse_app_index[user_entries[i]]
        return user_entries
    
    def load_mappings(self):
        '''
        Loads mappings for user_id and app_id to index and reverse mappings. 
        Saves the mappings to files if they don't exist using joblib for serialization.
        '''
        import os
        import joblib
        mapping_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'funk-svd', 'mappings')
        user_index_file = os.path.join(mapping_dir, 'user_index.joblib')
        app_index_file = os.path.join(mapping_dir, 'app_index.joblib')
        reverse_user_index_file = os.path.join(mapping_dir, 'reverse_user_index.joblib')
        reverse_app_index_file = os.path.join(mapping_dir, 'reverse_app_index.joblib')

        os.makedirs(mapping_dir, exist_ok=True)

        if (os.path.exists(user_index_file) and 
            os.path.exists(app_index_file) and 
            os.path.exists(reverse_user_index_file) and 
            os.path.exists(reverse_app_index_file)):
            self.user_index = joblib.load(user_index_file)
            self.app_index = joblib.load(app_index_file)
            self.reverse_user_index = joblib.load(reverse_user_index_file)
            self.reverse_app_index = joblib.load(reverse_app_index_file)
        else:
            if self.games is None:
                self.games = pd.read_csv(os.path.join(self.data_directory, 'games.csv'))
            if self.users is None:
                self.users = pd.read_csv(os.path.join(self.data_directory, 'users.csv'))

            unique_userid = self.users['user_id'].unique()
            unique_appid = self.games['app_id'].unique()

            self.user_index = {user_id: idx for idx, user_id in enumerate(unique_userid)}
            self.app_index = {app_id: idx for idx, app_id in enumerate(unique_appid)}
            self.reverse_user_index = {idx: user_id for idx, user_id in enumerate(unique_userid)}
            self.reverse_app_index = {idx: app_id for idx, app_id in enumerate(unique_appid)}

            joblib.dump(self.user_index, user_index_file)
            joblib.dump(self.app_index, app_index_file)
            joblib.dump(self.reverse_user_index, reverse_user_index_file)
            joblib.dump(self.reverse_app_index, reverse_app_index_file)


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

def get_user_test_ids(matrix_file_path: str):
    """
    Loads a sparse matrix from an .npz file and returns a set of unique user IDs (row indices).

    Args:
        matrix_file_path (str): The path to the .npz file.

    Returns:
        set: A set containing the unique user IDs present in the matrix, or None if there was an error loading the file.
    """
    try:
        matrix = load_npz(matrix_file_path)
        user_ids = set(matrix.row)  
        return user_ids
    except FileNotFoundError:
        print(f"Error: File not found at {matrix_file_path}")
        return None
    except Exception as e:  
        print(f"An error occurred while loading the matrix: {e}")
        return None

def load_all_items_id(items_path):
    """
    Loads item data from a CSV file, extracts unique app IDs, and returns the count.

    Args:
        items_path (str): The path to the items CSV file.

    Returns:
        int or None: The number of unique app IDs, or None if there's an error.
    """
    try:
        items_df = pd.read_csv(items_path)
        if 'app_id' not in items_df.columns:
            print(f"Error: 'app_id' column not found in {items_path}")
            return None

        all_items = items_df['app_id'].unique()
        return all_items

    except FileNotFoundError:
        print(f"Error: File not found at {items_path}")
        return None
    except pd.errors.ParserError:  # Handle CSV parsing errors
        print(f"Error: Could not parse CSV file at {items_path}. Check file format.")
        return None
    except Exception as e:  # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        return None

def funk_svd_testing():
    import importlib.util
    from pathlib import Path

    module_path = Path("funk-svd/model.py")
    spec = importlib.util.spec_from_file_location("model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Testing = module.Testing

    test_instance = Testing('./', 'train_and_test.npz')

    test_user_ids = get_user_test_ids('test_matrix.npz')
    history_getter = HistoryGetter('rest_test.npz')

    all_items = load_all_items_id('games.csv')

    predicted_list = []
    actual_list = []
    exit_limit = 1000
    exit_count = 0
    for user_id in tqdm.tqdm(test_user_ids, total=exit_limit, desc="Processing Users"):
        exit_count += 1
        predicted = test_instance.ask_for_recommendation(user_id, 20)
        new_predicted = []

        for item in predicted:  # Iterate directly over the dictionaries in the list
            new_predicted.append(item['app_id']) #access app_id from current item

        predicted = new_predicted
        actual = history_getter.get_user_actual(user_id)

        predicted_list.append(predicted)
        actual_list.append(actual)
        if(exit_count == exit_limit):
            break

    test_results = test_model(predicted_list, actual_list, k=20, all_items=all_items)
    print("Funk svd metrics")
    print(test_results)

load_all_items_id('games.csv')[0]
funk_svd_testing()

# print(load_all_items_id('games.csv'))

# abc = HistoryGetter('rest_test.npz')
# abc.build_user_histories()
# print(abc.get_user_actual(5))
# print(abc.get_user_actual(12))
# print(abc.get_user_actual(13))


