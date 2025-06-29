# Movie_Recommendation_System

## Overview
This project is a web-based hybrid movie recommendation system that combines **Collaborative Filtering (CF)** and **Content-Based Filtering (CBF)** to suggest movies tailored to a user's taste. It also integrates the **TMDb API** to fetch real-time movie posters for a visually engaging experience. The app is built with **Streamlit** and allows users to dynamically adjust the blend between CF and CBF via an alpha slider.

## Dataset
The recommendation engine uses the [MovieLens 1M dataset] which includes:
- 1 million ratings from 6,000 users on 4,000 movies.
- Metadata such as movie genres and user ratings.
- User IDs, Movie IDs, and Timestamps.

## Steps
- The dataset is preprocessed to create a **user-item matrix** for CF and a **genre-based TF-IDF matrix** for CBF.
- The CF model uses **k-Nearest Neighbors** (KNN with cosine similarity) to identify similar users and their favorite movies.
- The CBF model uses **cosine similarity on TF-IDF vectors** of genres to find similar movies.
- A **hybrid scoring mechanism** blends CF and CBF scores, allowing users to set the importance of each using an alpha slider.
- **TMDb API** is called to fetch movie posters dynamically.
- Users can view:
  - Recently rated movies with posters
  - User insights like top genres and rating distribution
  - Top 10 globally most-rated movies

## Features
- Interactive Streamlit interface
- TMDb poster integration
- Adjustable alpha blending (0 = CF only, 1 = CBF only)
- Genre filtering
- User insights and visualizations (bar and pie charts)
- Modular code separated into `app.py` and `recommender.py`


