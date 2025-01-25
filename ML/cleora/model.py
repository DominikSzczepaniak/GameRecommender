import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycleora import SparseMatrix
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors


class CleoraRecommender:

    def __init__(self, model_dir="cleora_model3", dim=128, iter=3):
        self.model_dir = Path(model_dir)
        self.dim = dim
        self.iter = iter
        self.item_embeddings = None
        self.user_to_items = None
        self.item_id_to_idx = {}
        self.loaded = False

    def fit(self, data_path):
        """Process NPZ file through DataFrame conversion"""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Get DataFrame from sparse matrix
        df = self.convert_sparse_to_csv(data_path)

        # Convert to user-item groups (filter positive ratings)
        user_groups = defaultdict(list)
        for _, row in df[df["rating"] == 1].iterrows():
            user_groups[row["u_id"]].append(f"item_{row['i_id']}")

        # Create generator instead of list
        cleora_input = (" ".join(items) for items in user_groups.values())

        mat = SparseMatrix.from_iterator(cleora_input,
                                         columns="complex::reflexive::product")

        embeddings = mat.initialize_deterministically(self.dim)
        for _ in range(self.iter):
            embeddings = mat.left_markov_propagate(embeddings)
            embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        # Save model
        self.item_ids = mat.entity_ids
        self.item_embeddings = embeddings
        self.user_to_items = {
            u: set(items)
            for u, items in user_groups.items()
        }
        self.item_id_to_idx = {
            item: idx
            for idx, item in enumerate(self.item_ids)
        }

        # Train KNN once
        self.knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn_model.fit(self.item_embeddings)

        self._save_model()

    def convert_sparse_to_csv(self, path):
        """Convert sparse matrix to DataFrame with (user, item, rating) tuples"""
        sparse_matrix = load_npz(path)
        rows, cols = sparse_matrix.row, sparse_matrix.col
        return pd.DataFrame({
            "u_id": rows,
            "i_id": cols,
            "rating": sparse_matrix.data
        })

    def _save_model(self):
        """Save model with KNN"""
        with open(self.model_dir / "embeddings.npy", "wb") as f:
            np.save(f, self.item_embeddings)
        with open(self.model_dir / "mappings.pkl", "wb") as f:
            pickle.dump(
                {
                    "item_ids": self.item_ids,
                    "item_id_to_idx": self.item_id_to_idx,
                    "user_to_items": self.user_to_items,
                },
                f,
            )

        # Save KNN model
        with open(self.model_dir / "knn_model.pkl", "wb") as f:
            pickle.dump(self.knn_model, f)

    def load_model(self):
        """Load pre-trained model"""
        self.loaded = True
        with open(self.model_dir / "embeddings.npy", "rb") as f:
            self.item_embeddings = np.load(f)
        with open(self.model_dir / "mappings.pkl", "rb") as f:
            data = pickle.load(f)
            self.item_ids = data["item_ids"]
            self.item_id_to_idx = data["item_id_to_idx"]
            self.user_to_items = data["user_to_items"]

        with open(self.model_dir / "knn_model.pkl", "rb") as f:
            self.knn_model = pickle.load(f)

    def ask_for_recommendation(self, user_id, k=10):
        """Get recommendations using pre-trained KNN"""
        if not self.loaded:
            self.load_model()

        seen_items = self.user_to_items.get(user_id, set())
        if not seen_items:
            return []

        # Get user embedding (average of seen items)
        seen_indices = [
            self.item_id_to_idx[item] for item in seen_items
            if item in self.item_id_to_idx
        ]
        if not seen_indices:
            return []

        user_embedding = self.item_embeddings[seen_indices].mean(axis=0)
        user_embedding = user_embedding.reshape(1, -1)

        # Query KNN
        distances, indices = self.knn_model.kneighbors(
            user_embedding,
            n_neighbors=k +
            len(seen_indices),  # Get extra to filter seen items
        )

        # Filter seen items and return top k
        recommendations = []
        for idx in indices[0]:
            item_id = self.item_ids[idx]
            if item_id not in seen_items:
                recommendations.append(int(item_id.split("_")[1]))
                if len(recommendations) >= k:
                    break

        return recommendations


if __name__ == "__main__":
    recommender = CleoraRecommender(dim=4096, iter=8)
    recommender.fit("./data/train_and_test.npz")
