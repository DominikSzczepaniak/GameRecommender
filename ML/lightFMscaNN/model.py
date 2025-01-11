import numpy as np
import pandas as pd
import scann
from lightfm import LightFM
from scipy.sparse import coo_matrix, save_npz, load_npz


class LightFMscaNN:
  def __init__(self):
    '''Initialize the LightFMscaNN model'''

    try:
      self.usersM = pd.read_csv('./data/users.csv')
      self.gamesM = pd.read_csv('./data/games.csv')
      self.recommendationsM = pd.read_csv('./data/recommendations.csv')
      self.USERxGAME = load_npz('./data/rating_matrix_sparse.npz').tocsr()
      self.model = self.loadModel()
    except:
      raise ImportError('Model initialization failed. Please check if the data files are present.')
  
  def fit(self, interaction, epochs=10):
    '''Fit the LightFM model to create embeddings'''
    
    for epoch in range(epochs):
      self.model.fit_partial(self.USERxGAME, epochs=1, num_threads=4)

      itemEmbeddings = self.model.item_embeddings
      userEmbeddings = self.model.user_embeddings
      itemBiases = self.model.item_biases
      userBiases = self.model.user_biases

      print(f'Epoch {epoch} completed!')
      
      np.save('./data/model/item_embeddings.npy', itemEmbeddings)
      np.save('./data/model/user_embeddings.npy', userEmbeddings)
      np.save('./data/model/item_biases.npy', itemBiases)
      np.save('./data/model/user_biases.npy', userBiases)


  def loadModel(self):
    '''
      Loads the pre-trained model

      Returns:
        model: LightFM class object
    '''
    # Load the pre-trained model
    itemEmbeddings = np.load('./data/model/item_embeddings.npy')
    userEmbeddings = np.load('./data/model/user_embeddings.npy')
    itemBiases = np.load('./data/model/item_biases.npy')
    userBiases = np.load('./data/model/user_biases.npy')

    # Initialize the model
    model = LightFM(learning_schedule='adagrad', loss='warp')

    model.item_embeddings = itemEmbeddings
    model.user_embeddings = userEmbeddings
    model.item_biases = itemBiases
    model.user_biases = userBiases

    return model
