# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import requests
import re

# --- Page config ---
st.set_page_config(layout="centered")

# --- TMDb API Key ---
TMDB_API_KEY = "0f69e89b1f60756123aaaad6b5e448de"

# --- Data Loading ---
@st.cache_data

def load_data():
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["MovieID", "Title", "Genres"])
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python", encoding="latin-1",
                          names=["UserID", "MovieID", "Rating", "Timestamp"])
    return movies, ratings

movies, ratings = load_data()

# --- Content-Based Filtering ---
tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
tfidf_matrix = tfidf.fit_transform(movies['Genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

# --- Collaborative Filtering ---
user_item = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item.values)

# Predict CF scores for all movies for a given user
def get_cf_scores(user_id):
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

# Clean titles for TMDb

def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

# Fetch movie poster from TMDb
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

# Hybrid recommendation
def hybrid_recommend(user_id, movie_title, alpha=0.5, top_n=10):
    if movie_title not in movies['Title'].values:
        return pd.DataFrame(columns=['Title', 'Genres', 'Poster'])

    idx = movies[movies['Title'] == movie_title].index[0]
    cb_scores = pd.Series(cosine_sim[idx], index=movies.index)
    cf_scores = get_cf_scores(user_id)

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

# Render movie card
def render_movie_card(poster, title, genres):
    return f"""
    <div style='text-align:center; padding:10px'>
        <img src='{poster}' style='width:150px; height:auto; border-radius:10px'><br>
        <b style='font-size:14px'>{title}</b><br>
        <div style='font-size:12px; color:gray; margin-top:4px'>{genres}</div>
    </div>
    """

# --- Streamlit UI ---
st.title("🎬 Hybrid Movie Recommendation System")

valid_users = ratings['UserID'].unique()
user_id = st.number_input("👤 Enter User ID:", min_value=int(min(valid_users)), 
                          max_value=int(max(valid_users)), value=1, step=1)

recent_rated = ratings[ratings['UserID'] == user_id].merge(movies, on='MovieID')
recent_rated = recent_rated.sort_values(by='Timestamp', ascending=False).head(5)
st.markdown("**🎥 Recently Rated Movies:**")
row_count = 3
for i in range(0, len(recent_rated), row_count):
    row = recent_rated.iloc[i:i+row_count]
    cols = st.columns(row_count)
    for j, (_, movie) in enumerate(row.iterrows()):
        with cols[j]:
            poster = get_movie_poster(movie['Title'])
            st.markdown(render_movie_card(poster, movie['Title'], movie['Genres']), unsafe_allow_html=True)

with st.expander("📊 User Insights"):
    genre_data = recent_rated['Genres'].str.split('|').explode().value_counts().head(10)
    fig1, ax1 = plt.subplots()
    genre_data.plot(kind='bar', ax=ax1)
    ax1.set_title("Top Genres Rated by User")
    st.pyplot(fig1)

    rating_dist = ratings[ratings['UserID'] == user_id]['Rating'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.pie(rating_dist, labels=rating_dist.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title("User Rating Distribution")
    st.pyplot(fig2)

    st.markdown("---")
    top_movies = ratings['MovieID'].value_counts().head(10).index
    top_titles = movies[movies['MovieID'].isin(top_movies)].merge(
        ratings.groupby('MovieID').count()['Rating'], on='MovieID')
    fig3, ax3 = plt.subplots()
    sns.barplot(data=top_titles, y='Title', x='Rating', ax=ax3)
    ax3.set_title("Top 10 Most Rated Movies (All Users)")
    st.pyplot(fig3)

movie_title = st.selectbox("🎞️ Pick a movie you liked:", sorted(movies['Title'].unique()))
strategy = st.radio("🎯 Strategy:", ['Hybrid', 'Content-Based Only', 'Collaborative Only'])
selected_genre = st.selectbox("🎭 Filter results by genre (optional):", ['All'] + sorted(tfidf.get_feature_names_out()))

alpha = {
    'Hybrid': st.slider("⚖️ Blend (Content vs CF):", 0.0, 1.0, 0.5, 0.1),
    'Content-Based Only': 1.0,
    'Collaborative Only': 0.0
}[strategy]

if st.button("🔍 Get Recommendations"):
    results = hybrid_recommend(user_id, movie_title, alpha=alpha, top_n=10)

    if selected_genre != 'All':
        results = results[results['Genres'].str.contains(selected_genre)]

    if results.empty:
        st.warning("No recommendations found. Try a different movie or strategy.")
    else:
        st.subheader("✅ Recommended Movies")
        for i in range(0, len(results), row_count):
            row = results.iloc[i:i+row_count]
            cols = st.columns(row_count)
            for j, (_, movie) in enumerate(row.iterrows()):
                with cols[j]:
                    st.markdown(render_movie_card(movie['Poster'], movie['Title'], movie['Genres']), unsafe_allow_html=True)
