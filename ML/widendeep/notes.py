#!/usr/bin/env python3
# train_wide_and_deep_bpr.py

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.sparse import load_npz
from math import ceil

##############################################################################
# 1. Parametry konfiguracyjne
##############################################################################
NPZ_FILE = "./data/rating_matrix_sparse.npz"
BATCH_SIZE = 4096
EPOCHS = 2
EMBEDDING_DIM = 32
MODEL_FILENAME = "bpr_model.h5"
SUBMODEL_FILENAME = "score_model.h5"

##############################################################################
# 2. Wczytanie macierzy i konwersja do CSR
##############################################################################
sparse_mat = load_npz(NPZ_FILE)
csr_mat = sparse_mat.tocsr()
num_users, num_items = csr_mat.shape

print("[INFO] Wczytano macierz z:", NPZ_FILE)
print("       Kształt (num_users, num_items):", csr_mat.shape)
print("       Niezerowych interakcji (1):", csr_mat.nnz)

##############################################################################
# 3. Generator pairwise (BPR): zwraca (user, pos_item, neg_item)
##############################################################################
class BprInteractionGenerator(keras.utils.Sequence):
    """
    Generator Keras, który dla każdej pozytywnej interakcji (u, i) wybiera
    1 negatywny przykład (j). Zwraca batch (u, i, j).
    Dodatkowo zwracamy dummy_labels, bo Keras wymaga (X, y).
    """

    def __init__(self, csr_matrix, batch_size=1024, shuffle_users=True, **kwargs):
        super().__init__(**kwargs)

        self.csr_mat = csr_matrix
        self.num_users, self.num_items = csr_matrix.shape
        self.batch_size = batch_size
        self.shuffle_users = shuffle_users

        self.users = np.arange(self.num_users, dtype=np.int32)
        if shuffle_users:
            np.random.shuffle(self.users)

        self.total_positives = self.csr_mat.nnz

        self.steps_per_epoch = ceil(self.total_positives / self.batch_size)

        self._user_idx = 0

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_user = []
        batch_pos_item = []
        batch_neg_item = []

        while len(batch_user) < self.batch_size:
            if self._user_idx >= self.num_users:
                self._user_idx = 0
                if self.shuffle_users:
                    np.random.shuffle(self.users)

            user = self.users[self._user_idx]
            self._user_idx += 1

            row_start = self.csr_mat.indptr[user]
            row_end = self.csr_mat.indptr[user+1]
            pos_items = self.csr_mat.indices[row_start:row_end]

            for pos_item in pos_items:
                neg_item = random.randint(0, self.num_items - 1)
                while neg_item in pos_items:
                    neg_item = random.randint(0, self.num_items - 1)

                batch_user.append(user)
                batch_pos_item.append(pos_item)
                batch_neg_item.append(neg_item)

                if len(batch_user) >= self.batch_size:
                    break

        batch_user = batch_user[:self.batch_size]
        batch_pos_item = batch_pos_item[:self.batch_size]
        batch_neg_item = batch_neg_item[:self.batch_size]

        dummy_y = np.zeros(len(batch_user), dtype=np.float32)

        X = {
            "user_id": np.array(batch_user, dtype=np.int32),
            "pos_item_id": np.array(batch_pos_item, dtype=np.int32),
            "neg_item_id": np.array(batch_neg_item, dtype=np.int32),
        }
        return (X, dummy_y)

    def on_epoch_end(self):
        self._user_idx = 0
        if self.shuffle_users:
            np.random.shuffle(self.users)


##############################################################################
# 4. Model "score_model" liczący score(u, item) w stylu wide & deep
#    + Model BPR, który wczytuje (user, pos_item, neg_item) i zwraca [pos_score, neg_score]
##############################################################################

user_input = keras.Input(shape=(1,), name="user_id")
item_input = keras.Input(shape=(1,), name="item_id")

user_embedding_wide = layers.Embedding(
    input_dim=num_users,
    output_dim=1,
    input_length=1
)(user_input)
user_embedding_wide = layers.Flatten()(user_embedding_wide)

item_embedding_wide = layers.Embedding(
    input_dim=num_items,
    output_dim=1,
    input_length=1
)(item_input)
item_embedding_wide = layers.Flatten()(item_embedding_wide)

wide_part = layers.Concatenate()([user_embedding_wide, item_embedding_wide])
wide_part = layers.Dense(1, activation='linear', name='wide_linear')(wide_part)

embedding_dim = EMBEDDING_DIM

user_embedding_deep = layers.Embedding(
    input_dim=num_users,
    output_dim=embedding_dim,
    input_length=1,
    name='user_embedding_deep'
)(user_input)
user_embedding_deep = layers.Flatten()(user_embedding_deep)

item_embedding_deep = layers.Embedding(
    input_dim=num_items,
    output_dim=embedding_dim,
    input_length=1,
    name='game_embedding_deep'
)(item_input)
item_embedding_deep = layers.Flatten()(item_embedding_deep)

deep_concat = layers.Concatenate()([user_embedding_deep, item_embedding_deep])
deep = layers.Dense(64, activation='relu')(deep_concat)
deep = layers.Dense(32, activation='relu')(deep)
deep = layers.Dense(16, activation='relu')(deep)
deep_part = layers.Dense(1, activation='linear', name='deep_linear')(deep)

score_out = layers.Add()([wide_part, deep_part])

score_model = keras.Model(
    inputs=[user_input, item_input],
    outputs=score_out,
    name="score_model"
)

user_bpr_input = keras.Input(shape=(1,), name="user_id")
pos_item_bpr_input = keras.Input(shape=(1,), name="pos_item_id")
neg_item_bpr_input = keras.Input(shape=(1,), name="neg_item_id")

score_pos = score_model([user_bpr_input, pos_item_bpr_input])
score_neg = score_model([user_bpr_input, neg_item_bpr_input])

bpr_output = layers.Concatenate(axis=1)([score_pos, score_neg])

bpr_model = keras.Model(
    inputs=[user_bpr_input, pos_item_bpr_input, neg_item_bpr_input],
    outputs=bpr_output,
    name="bpr_model"
)


##############################################################################
# 5. Custom loss: BPR
#    L_BPR = - 1/N sum(log( sigma(score_pos - score_neg) ))
##############################################################################
def bpr_loss(y_true, y_pred):
    """
    y_pred ma shape (batch, 2): kolumny [score_pos, score_neg].
    y_true jest tu dummy, bo BPR pairwise nie potrzebuje 0/1.
    """
    score_pos = y_pred[:, 0]
    score_neg = y_pred[:, 1]
    return -tf.reduce_mean(tf.math.log_sigmoid(score_pos - score_neg))


bpr_model.compile(
    optimizer='adam',
    loss=bpr_loss
)

bpr_model.summary()

##############################################################################
# 6. Trenowanie modelu BPR
##############################################################################

train_gen = BprInteractionGenerator(
    csr_matrix=csr_mat,
    batch_size=BATCH_SIZE,
    shuffle_users=True
)

steps_per_epoch = len(train_gen)
print(f"[INFO] steps_per_epoch = {steps_per_epoch}")

bpr_model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    verbose=1
)

##############################################################################
# 7. Zapis wytrenowanego modelu BPR
#    Uwaga: Zapisujemy "bpr_model" - to para (pos_score, neg_score).
#    Do rekomendacji top-N przyda nam się "score_model".
##############################################################################
bpr_model.save(MODEL_FILENAME)
score_model.save(SUBMODEL_FILENAME)
print(f"[INFO] Model BPR zapisany do pliku '{MODEL_FILENAME}'")
print(f"[INFO] Model score zapisany do pliku '{SUBMODEL_FILENAME}'")

print("\n[INFO] Zakończono działanie skryptu BPR wide & deep.")
