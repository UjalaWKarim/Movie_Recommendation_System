# recommender.py
import pandas as pd
import numpy as np
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

TMDB_API_KEY = "0f69e89b1f60756123aaaad6b5e448de"

def load_data():
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["MovieID", "Title", "Genres"])
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python", encoding="latin-1",
                          names=["UserID", "MovieID", "Rating", "Timestamp"])
    return movies, ratings

def train_cbf(movies):
    tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
    tfidf_matrix = tfidf.fit_transform(movies['Genres'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return tfidf, cosine_sim

def train_cf(ratings):
    user_item = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item.values)
    return user_item, model_knn

def get_cf_scores(user_id, user_item, model_knn):
    if user_id not in user_item.index:
        return pd.Series(dtype='float64')

    user_vec = user_item.loc[user_id].values.reshape(1, -1)
    dists, inds = model_knn.kneighbors(user_vec, n_neighbors=11)
    similar_users = user_item.index[inds.flatten()[1:]]
    sim_weights = 1 - dists.flatten()[1:]
    sim_weights /= sim_weights.sum()

    neighbor_ratings = user_item.loc[similar_users]
    weighted_ratings = neighbor_ratings.T.dot(sim_weights)
    return weighted_ratings

def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

def get_movie_poster(title):
    title = clean_title(title)
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for movie in data['results']:
                poster_path = movie.get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://upload.wikimedia.org/wikipedia/commons/f/fc/No_picture_available.png"

def hybrid_recommend(user_id, movie_title, alpha, top_n, movies, ratings, user_item, model_knn, cosine_sim):
    if movie_title not in movies['Title'].values:
        return pd.DataFrame(columns=['Title', 'Genres', 'Poster'])

    idx = movies[movies['Title'] == movie_title].index[0]
    cb_scores = pd.Series(cosine_sim[idx], index=movies.index)
    cf_scores = get_cf_scores(user_id, user_item, model_knn)

    combined_scores = []
    for movie_id in movies['MovieID']:
        cb_score = cb_scores[movies[movies['MovieID'] == movie_id].index[0]] if movie_id in movies['MovieID'].values else 0
        cf_score = cf_scores.get(movie_id, 0)
        hybrid_score = alpha * cb_score + (1 - alpha) * cf_score
        combined_scores.append((movie_id, hybrid_score))

    top_ids = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]
    top_movies = movies[movies['MovieID'].isin([x[0] for x in top_ids])]

    results = []
    for _, row in top_movies.iterrows():
        results.append({
            'Title': row['Title'],
            'Genres': row['Genres'],
            'Poster': get_movie_poster(row['Title'])
        })
    return pd.DataFrame(results)
