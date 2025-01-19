import pandas as pd
from scipy.sparse import load_npz
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