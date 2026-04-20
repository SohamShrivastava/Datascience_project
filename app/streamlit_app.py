import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os

# FIX PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cold_start.cold_start import ColdStartRecommender
from explain.explain import Explainer

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- LOAD ----------------
@st.cache_resource
def load_models():
    with open("outputs/saved_models/mf.pkl", "rb") as f:
        mf = pickle.load(f)

    with open("outputs/saved_models/preprocessor.pkl", "rb") as f:
        pre = pickle.load(f)

    with open("outputs/saved_models/movies.pkl", "rb") as f:
        movies = pickle.load(f)

    return mf, pre, movies


mf, pre, movies = load_models()
ratings = pd.read_csv("data/ratings.csv")

df = pd.merge(ratings, movies, on="movieId")

# 🔥 ADD THIS LINE HERE
df = pre.encode_ids(df)

# ---------------- TITLE ----------------
st.title("🎬 Movie Recommender System")
st.markdown("### 🚀 Data Science Project")

# ---------------- SIDEBAR ----------------
page = st.sidebar.selectbox(
    "Choose Page",
    ["Dashboard", "Recommend", "Cold Start", "Explain", "Model Comparison"]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.header("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Users", df['userId'].nunique())
    col2.metric("Movies", df['movieId'].nunique())
    col3.metric("Ratings", len(df))

    st.subheader("Top Popular Movies")
    popular = ratings.groupby("movieId").size().reset_index(name="count")
    popular = popular.merge(movies, on="movieId")
    popular = popular.sort_values(by="count", ascending=False).head(10)

    st.dataframe(popular[['title', 'count']])

# ---------------- RECOMMEND ----------------
elif page == "Recommend":
    st.header("🎥 Personalized Movie Recommendations")

    user_list = sorted(df['user'].unique())
    user_id = st.selectbox("Select User", user_list)

    if st.button("Get Recommendations"):

        st.markdown(f"### 🍿 Recommendations for User {user_id}")

        # explanation (top part)
        st.markdown("""
        💡 **How this works:**

        - We look at movies this user has liked in the past  
        - Find other users with similar taste  
        - Predict how much this user will like each movie  
        - Recommend the top movies with highest predicted ratings  
        """)

        scores = []

        for item in range(len(pre.movie_encoder.classes_)):
            score = mf.predict(user_id, item)
            scores.append((item, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]

        st.markdown("---")

        for item, score in scores:
            movieId = pre.movie_encoder.inverse_transform([item])[0]
            movie_row = movies[movies['movieId'] == movieId]

            if movie_row.empty:
                continue

            title = movie_row['title'].values[0]
            genres = movie_row['genres'].values[0]

            # interpretation
            if score >= 4:
                label = "🔥 Strong recommendation"
            elif score >= 3:
                label = "👍 Moderate recommendation"
            else:
                label = "❌ Weak recommendation"

            st.markdown(f"""
### 🎬 {title}

⭐ **Predicted Rating:** {round(score, 2)} / 5  
👉 **{label}**

📂 **Genres:** {genres}

---
""")

        # bottom explanation
        st.markdown("""
### 🧠 What does the rating mean?

- ⭐ **4.0+** → You will very likely enjoy this  
- ⭐ **3.0 - 4.0** → You might like it  
- ⭐ **Below 3.0** → Probably not your taste  

---

### 🎯 In simple words:

We studied your past preferences, compared you with similar users,  
and predicted which movies you would enjoy the most.
""")

# ---------------- COLD START ----------------
elif page == "Cold Start":
    st.header("🧊 Cold Start (New User)")

    mode = st.radio("Choose Mode", ["Genre Based", "Movie Based"])

    cold = ColdStartRecommender(movies, ratings)

    # -------- GENRE BASED --------
    if mode == "Genre Based":
        all_genres = list(set("|".join(movies['genres']).split("|")))
        selected = st.multiselect("Select Genres", all_genres)

        if st.button("Recommend"):
            res = cold.recommend(selected)
            st.dataframe(res)

    # -------- MOVIE BASED --------
    else:
        movie_list = movies['title'].dropna().unique()

        # OPTIONAL: limit list size (faster UI)
        movie_list = sorted(movie_list)[:1000]

        selected_movies = st.multiselect("Select Movies You Like", movie_list)

        if st.button("Recommend"):
            res = cold.recommend_from_movies(selected_movies)
            st.dataframe(res)

# ---------------- EXPLAIN ----------------
elif page == "Explain":
    st.header("🧠 Explain Recommendation")

    # ---------- USER SELECT ----------
    user_id = st.selectbox(
        "Select User",
        sorted(df['user'].unique()),
        format_func=lambda x: f"User {x}"
    )

    # ---------- MOVIE SELECT ----------
    # movie_list = movies['title'].dropna().unique()
    # selected_movie = st.selectbox("Select Movie", sorted(movie_list))

    valid_movieIds = set(pre.movie_encoder.classes_)

    valid_movies = movies[movies['movieId'].isin(valid_movieIds)]

    movie_list = valid_movies['title'].dropna().unique()

    selected_movie = st.selectbox("Select Movie", sorted(movie_list))
    
    # ---------- BUTTON ----------
    if st.button("Explain"):

        # convert movie → encoded item
        movieId = movies[movies['title'] == selected_movie]['movieId'].values[0]
        if movieId not in pre.movie_encoder.classes_:
            st.error("😕 Sorry, we can’t analyze this movie yet. Please try another one.")
            st.stop()
        item_id = pre.movie_encoder.transform([movieId])[0]

        # ---------- USER HISTORY ----------
        user_data = df[df['user'] == user_id]
        liked = user_data.sort_values(by='rating', ascending=False).head(5)
        liked_movies = liked[['title', 'rating', 'genres']]

        # ---------- RUN EXPLAINER ----------
        explainer = Explainer(df, movies, mf)
        explanation = explainer.explain(user_id, item_id, pre)

        # ---------- EXTRACT RATING ----------
        import re
        match = re.search(r"Predicted rating.*: ([0-9.]+)", explanation)
        rating = float(match.group(1)) if match else 0

        # ---------- INTERPRET ----------
        if rating >= 4:
            strength = "🔥 Strong match"
            verdict = f"YES, User {user_id} will likely LOVE this movie"
        elif rating >= 3:
            strength = "👍 Moderate match"
            verdict = f"YES, User {user_id} may like this movie"
        else:
            strength = "❌ Weak match"
            verdict = f"NOT recommended for User {user_id}"

        # ---------- MAIN OUTPUT ----------
        st.markdown(f"""
        ## 🎬 Will User {user_id} like {selected_movie}?

        ### {verdict}
        """)

        st.markdown("---")

        # ---------- USER HISTORY ----------
        st.markdown(f"### 👉 1. Based on User {user_id}'s past behavior")

        if len(liked_movies) > 0:
            for _, row in liked_movies.iterrows():
                st.write(f"• {row['title']} ⭐ {row['rating']}")
        else:
            st.write(f"No past ratings found for User {user_id}")

        # ---------- GENRE MATCH ----------
        st.markdown("### 👉 2. Similarity (Genres Match)")

        selected_genres = movies[movies['movieId'] == movieId]['genres'].values[0]
        st.write(f"Selected movie genres: **{selected_genres}**")

        st.write("These genres overlap with your liked movies")

        # ---------- MODEL OUTPUT ----------
        st.markdown("### 👉 3. Model Prediction (Matrix Factorization)")
        st.write(f"Predicted rating: **{round(rating,2)} / 5**")
        st.write(f"Interpretation: **{strength}**")
        st.markdown("""
        💡 **What does this rating mean?**

        - 🔥 **4.0 – 5.0** → Strong recommendation (you will likely love it)
        - 👍 **3.0 – 4.0** → Moderate recommendation (you may like it)
        - ❌ **Below 3.0** → Weak recommendation (not a good match)
        """)

        st.markdown("---")

        # ---------- SIMPLE EXPLANATION ----------
        st.markdown(f"""
📌 **Simple Explanation**

We looked at movies **User {user_id} liked before**,  
found this movie is similar,  
and predicted whether **User {user_id} would enjoy it**.
""")

        # ---------- TECH DETAILS ----------
        with st.expander("🔍 Show detailed technical explanation"):
            st.text(explanation)





# ---------------- MODEL COMPARISON ----------------
elif page == "Model Comparison":
    st.header("📈 Model Performance")

    try:
        # RMSE/MAE table
        results = pd.read_csv("outputs/results.csv")
        st.subheader("Rating Prediction (RMSE / MAE)")
        st.dataframe(results)

        # Ranking metrics chart
        ranking = pd.read_csv("outputs/ranking_results.csv")
        st.subheader("Ranking Metrics")
        
        import plotly.express as px
        fig = px.bar(
            ranking,
            x="model",
            y=["Precision@10", "Recall@10"],
            barmode="group",
            title="Precision & Recall @10"
        )
        st.plotly_chart(fig)

        fig2 = px.bar(
            ranking,
            x="model", 
            y=["Diversity@10", "Novelty@10"],
            barmode="group",
            title="Diversity & Novelty @10"
        )
        st.plotly_chart(fig2)

    except FileNotFoundError:
        st.warning("Run evaluate_ranking.py first")