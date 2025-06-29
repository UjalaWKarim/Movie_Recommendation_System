# --- Imports ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from recommender import load_data, hybrid_recommend, get_movie_poster, render_movie_card, tfidf

# --- Page config ---
st.set_page_config(layout="centered")

# --- Data ---
movies, ratings = load_data()

# --- UI ---
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
