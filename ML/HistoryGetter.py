from scipy.sparse import load_npz
import pandas as pd

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