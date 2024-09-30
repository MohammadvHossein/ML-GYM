import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches
import pickle

movies_data = pd.read_csv('movies.csv')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
movies_data[selected_features] = movies_data[selected_features].fillna('')

combined_features = movies_data[selected_features].agg(' '.join, axis=1)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(feature_vectors)

with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn_model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('combined_features.pkl', 'wb') as features_file:
    pickle.dump(combined_features, features_file)

movies_data.to_csv('movies_data.csv', index=False)

print("Model, vectorizer, combined features, and data saved successfully.")