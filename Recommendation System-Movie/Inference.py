import pandas as pd
from difflib import get_close_matches
import pickle

with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('combined_features.pkl', 'rb') as features_file:
    combined_features = pickle.load(features_file)

movies_data = pd.read_csv('movies_data.csv')

movie_name = input('Enter your favorite movie name: ')
list_of_all_titles = movies_data['title'].tolist()

close_match = get_close_matches(movie_name, list_of_all_titles, n=1)

if close_match:
    close_match_title = close_match[0]
    close_match_index = list_of_all_titles.index(close_match_title)

    input_movie_vector = vectorizer.transform([combined_features[close_match_index]])

    distances, indices = knn_model.kneighbors(input_movie_vector, n_neighbors=6)

    print('Movies suggested for you:')
    for i in range(1, len(indices[0])):
        recommended_movie_index = indices[0][i]
        recommended_movie_title = movies_data.iloc[recommended_movie_index]['title']
        print(f"{i}. {recommended_movie_title}")
else:
    print("Movie not found in the database.")