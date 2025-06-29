import pandas as pd
import re
import requests
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

TMDB_API_KEY = "0f69e89b1f60756123aaaad6b5e448de"

def load_data():
    movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', encoding='latin-1',
                         names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python', encoding='latin-1',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    return movies, ratings

movies, ratings = load_data()

# Content-Based Filtering
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies['Genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Collaborative Filtering
user_item = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item.values)

def get_cf_recommend(user_id, top_n=50):
    if user_id not in user_item.index:
        return pd.DataFrame(columns=['MovieID', 'Title', 'Genres'])

    user_vec = user_item.loc[user_id].values.reshape(1, -1)
    dists, inds = model_knn.kneighbors(user_vec, n_neighbors=top_n+1)
    similar_users = user_item.index[inds.flatten()[1:]]
    rec_movies = ratings[ratings['UserID'].isin(similar_users)]
    top_movies = rec_movies.groupby('MovieID')['Rating'].mean().sort_values(ascending=False).head(top_n)
    return movies[movies['MovieID'].isin(top_movies.index)][['MovieID', 'Title', 'Genres']]

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

def hybrid_recommend(user_id, movie_title, top_n=10, alpha=0.5):
    if movie_title not in movies['Title'].values:
        return pd.DataFrame(columns=['Title', 'Genres', 'Poster'])

    idx = movies[movies['Title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+20]

    cf_recs = get_cf_recommend(user_id, top_n=50)
    cf_titles = set(cf_recs['Title'].values)

    hybrid_scores = []
    for i, sim in sim_scores:
        title = movies.iloc[i]['Title']
        match = get_close_matches(title, cf_titles, n=1, cutoff=0.95)
        cf_boost = 1 if match else 0
        hybrid_score = alpha * sim + (1 - alpha) * cf_boost
        hybrid_scores.append((i, hybrid_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for i, _ in hybrid_scores:
        row = movies.iloc[i][['Title', 'Genres']].to_dict()
        row['Poster'] = get_movie_poster(row['Title'])
        results.append(row)

    return pd.DataFrame(results)

def render_movie_card(poster, title, genres, rating=None):
    rating_html = f"<div style='font-size:13px; margin-top:4px'><b>⭐ Rating:</b> {rating}</div>" if rating is not None else ""
    return f"""
    <div style='text-align:center; padding:10px'>
        <img src='{poster}' style='width:150px; height:auto; border-radius:10px'><br>
        <b style='font-size:14px'>{title}</b><br>
        <div style='font-size:12px; color:gray; margin-top:4px'>{genres}</div>
        {rating_html}
    </div>
    """
