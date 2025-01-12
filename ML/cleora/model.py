def learn():
    from pycleora import SparseMatrix
    import numpy as np
    import pandas as pd
    from scipy.sparse import load_npz
    import joblib
    from sklearn.neighbors import NearestNeighbors

    recommendations = pd.read_csv('recommendations.csv')


    users_game = recommendations.groupby('user_id')['app_id'].apply(list).values

    cleora_input = map(lambda x: ' '.join(map(str, x)), users_game)

    mat = SparseMatrix.from_iterator(cleora_input, columns='complex::reflexive::app_id')
    embedding_dim = 4096
    embeddings = mat.initialize_deterministically(embedding_dim)

    NUM_WALKS = 2
    for i in range(NUM_WALKS):
        embeddings = mat.left_markov_propagate(embeddings)
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)

    joblib.dump(embeddings, 'game_embeddings.joblib')
    joblib.dump(mat.entity_ids, 'game_entity_ids.joblib')

    knn = NearestNeighbors(metric='euclidean', algorithm='auto')
    knn.fit(embeddings)
    joblib.dump(knn, 'knn_model.joblib')

def check():
    import joblib
    import numpy as np
    def recommend_similar_games(game_id, top_n=5):
        embeddings = joblib.load('game_embeddings.joblib')
        entity_ids = joblib.load('game_entity_ids.joblib')
        knn = joblib.load('knn_model.joblib')
        game_id = str(game_id)
        if game_id not in entity_ids:
            raise ValueError(f"Game ID '{game_id}' not found.")
        
        game_index = entity_ids.index(game_id)
        game_embedding = embeddings[game_index].reshape(1, -1)
        
        distances, indices = knn.kneighbors(game_embedding, n_neighbors=top_n + 1)
        similar_games = [entity_ids[i] for i in indices.flatten() if i != game_index]
        
        return similar_games, distances.flatten()

    game_id_to_recommend = 346110
    similar_games, distances = recommend_similar_games(game_id_to_recommend, top_n=37609)

    # Display recommended similar games
    print(f"Recommended similar games for Game ID {game_id_to_recommend}:")
    for game, dist in zip(similar_games, distances):
        print(f"Game ID: {game}, Similarity Distance: {dist:.4f}, game data: {get_game_data(game)}")

import pandas as pd
import sys
games = pd.read_csv('games.csv')
sys.stdout.reconfigure(encoding='utf-8')
def get_game_data(app_id):
    '''
    parameters: app_id - after mapping
    returns: game data (app_id, name, genres, etc.)
    '''
    app_id = int(app_id)
    result = games.loc[games['app_id'] == app_id]
    if result.empty:
        return None  
    return result.iloc[0]['title']
learn()
check()

#378648