import pickle

import numpy as np
import pandas as pd
import scann
from lightfm import LightFM
from lightfm.evaluation import recall_at_k
from scipy.sparse import coo_matrix, load_npz
from tqdm import tqdm


class Logger:

    @staticmethod
    def log(msg):
        print(f"[LOG] {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}")


class LightFMscaNN:
    # ----------------=[ Model initialization ]=----------------
    def __init__(self, k):
        try:
            self.users = pd.read_csv("./data/users.csv")
            self.games = pd.read_csv("./data/games.csv")
            self.recommendations = pd.read_csv("./data/recommendations.csv")
            self.games_metadata = pd.read_json("./data/games_metadata.json",
                                               lines=True)

            self.interactions = load_npz("./data/train_and_test.npz").tocsr()
            self.rest_test = load_npz("./data/rest_test.npz").tocsr()

            self.load_model('sixty4')

            unique_user_ids = self.users["user_id"].unique()
            unique_game_ids = self.games["app_id"].unique()

            self.user_id_to_index = {
                user_id: idx
                for idx, user_id in enumerate(unique_user_ids)
            }
            self.game_id_to_index = {
                game_id: idx
                for idx, game_id in enumerate(unique_game_ids)
            }
            self.index_to_user_id = {
                idx: user_id
                for user_id, idx in self.user_id_to_index.items()
            }
            self.index_to_game_id = {
                idx: game_id
                for game_id, idx in self.game_id_to_index.items()
            }
            self.game_id_to_title = {
                row["app_id"]: row["title"]
                for _, row in self.games.iterrows()
            }

            # # Fine tuned searcher
            # self.searcher = (scann.scann_ops_pybind.builder(
            #     self.model.item_embeddings, k,
            #     "dot_product").score_ah(6,
            #                             hash_type="lut256",
            #                             training_iterations=11).build())

            # Calculate popularity as interaction counts (train set only)
            train_popularity = np.array(
                self.interactions.sum(axis=0)).flatten()

            train_popularity += 1

            log_popularity = np.log(train_popularity)
            self.popularity_weights = (log_popularity - log_popularity.min()
                                       ) / (log_popularity.max() -
                                            log_popularity.min())

        except Exception as e:
            Logger.error(f"Model initialization failed: {e}")

    # ----------------=[ Model training ]=---------------
    def fit(self, name, epochs=100):
        for epoch in range(1, epochs + 1):
            self.model.fit_partial(self.interactions, epochs=5, num_threads=20)

            val_recall = recall_at_k(self.model,
                                     self.rest_test,
                                     k=20,
                                     num_threads=20).mean()

            print(f"Epoch {epoch}: Value of Recall@20 = {val_recall:.4f}")

            with open(f"./data/model/lightfm_{name}.pkl", "wb") as f:
                pickle.dump(model, f)

    def load_model(self, name):
        with open(f"./data/model/lightfm_{name}.pkl", "rb") as f:
            self.model: LightFM = pickle.load(f)

    # ----------------=[ Helper functions ]=---------------
    def list_user_liked_games(self, user_id):
        user_index = self.user_id_to_index[user_id]
        user_ratings = self.interactions[user_index].toarray()[0]

        games = []
        for idx, rating in enumerate(user_ratings):
            if rating == 1:
                games.append(self.index_to_game_id[idx])

        return games

    # -----------------=[ Prediction ]=------------------
    def embed_user(self, user_id, type=0):
        user_games = self.list_user_liked_games(user_id)

        if len(user_games) == 0:
            return np.zeros(64)

        if type == 0:
            game_indices = [
                self.game_id_to_index[game_id] for game_id in user_games
            ]
            game_embeddings = self.model.item_embeddings[game_indices]

            user_embedding = np.mean(game_embeddings, axis=0)

        elif type == 1:
            user_embedding = self.model.user_embeddings[
                self.user_id_to_index[user_id]]

        return user_embedding

    def predict(self, user_id, k, alpha=0.8):
        _, known_items = self.interactions[user_id].nonzero()

        all_items = np.arange(self.interactions.shape[1])
        candidate_items = np.setdiff1d(all_items, known_items)

        scores = self.model.predict(
            user_ids=np.full(len(candidate_items), user_id),
            item_ids=candidate_items,
            num_threads=20,
        )

        pop_scores = self.popularity_weights[candidate_items]

        combined_scores = alpha * scores + (1 - alpha) * pop_scores

        top_k_indices = np.argsort(-combined_scores)[:k]

        return candidate_items[top_k_indices]

    def predict2(self, user_id, k):
        user_embedding = self.embed_user(user_id)
        indices, scores = self.searcher.search(user_embedding)

        sorted_indices = np.argsort(-scores)
        sorted_item_indices = [indices[i] for i in sorted_indices]

        return sorted_item_indices[:k]

    # -----------------=[ Recommendation ]=-----------------
    def recommend(self, user_id, k):
        return self.predict(user_id, k)

    # -----------------=[ For Fun ]=------------------
    def similar_games(self, game_title, k):
        searcher = (scann.scann_ops_pybind.builder(
            self.model.item_embeddings, k, "dot_product").score_ah(1).build())

        game_id = self.games[self.games["title"] ==
                             game_title]["app_id"].values[0]

        game_index = self.game_id_to_index[game_id]
        game_embedding = self.model.item_embeddings[game_index]
        indices, scores = searcher.search(game_embedding)

        return [self.index_to_game_id[idx] for idx in indices]

    # -----------------=[ HYPER PARAMETERS ]=------------------

    def fine_tune(self):
        from hyperopt import STATUS_OK, hp

        space = {
            "no_components":
            hp.choice("no_components", [64, 100]),
            "loss":
            hp.choice("loss", ["warp", "warp-kos", "bpr"]),
            "learning_rate":
            hp.loguniform("learning_rate", np.log(1e-4), np.log(0.1)),
            "k":
            hp.choice("k", [10, 20]),
        }

        def objective(params):
            # Initialize model with sampled hyperparameters
            model = LightFM(
                no_components=params["no_components"],
                loss=params["loss"],
                k=params["k"],
                learning_rate=params["learning_rate"],
                random_state=42,
            )

            for _ in tqdm(range(30)):
                model.fit_partial(self.interactions, num_threads=20)

            val_recall = recall_at_k(model,
                                     self.rest_test,
                                     k=20,
                                     num_threads=20).mean()

            return {
                "loss": -val_recall,
                "status": STATUS_OK,
                "params": params,
            }

        from hyperopt import Trials, fmin, tpe

        trials = Trials()  # Track results
        best_params = fmin(
            fn=objective,  # Objective function
            space=space,  # Search space
            algo=tpe.
            suggest,  # Optimization algorithm (Tree-structured Parzen Estimator)
            max_evals=50,  # Number of trials (increase for better results)
            trials=trials,  # Store results
            verbose=True,  # Show progress
        )

        print("Best hyperparameters:", best_params)

        self.model = LightFM(
            no_components=best_params["no_components"],
            loss=best_params["loss"],
            k=best_params["k"],
            user_alpha=best_params["user_alpha"],
            item_alpha=best_params["item_alpha"],
            learning_rate=best_params["learning_rate"],
            random_state=42,
        )

        # Train longer (e.g., 100 epochs)
        self.fit(name="tuned_model", epochs=300)


if __name__ == "__main__":
    model = LightFMscaNN(10)

    model.fit("sixty6", 100)

    import sys

    sys.path.append("../")

    from metrics import *

    print(test_metrics(model, 20))
