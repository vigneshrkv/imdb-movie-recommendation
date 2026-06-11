"""
recommender.py
--------------
Core recommendation logic using TF-IDF and Cosine Similarity.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------- #
# Data Loading
# --------------------------------------------------------------------------- #

def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the IMDb movies CSV."""
    df = pd.read_csv(filepath)
    required_cols = {"Movie Name", "Storyline", "Cleaned_Storyline"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    df.dropna(subset=["Cleaned_Storyline"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------------------------------------------------------- #
# TF-IDF Model
# --------------------------------------------------------------------------- #

def build_tfidf_matrix(df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the Cleaned_Storyline column.

    Returns:
        vectorizer : fitted TfidfVectorizer
        tfidf_matrix : sparse matrix (n_movies x n_features)
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams for richer matching
        min_df=2,             # ignore very rare terms
    )
    tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Storyline"])
    return vectorizer, tfidf_matrix


# --------------------------------------------------------------------------- #
# Text Cleaning (for user input)
# --------------------------------------------------------------------------- #

def clean_input(text: str) -> str:
    """Apply the same cleaning steps used on the dataset."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove punctuation / numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #

def get_recommendations(
    user_input: str,
    df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Given a raw user storyline, return the top_n most similar movies.

    Returns a DataFrame with columns: Rank, Movie Name, Storyline, Similarity Score
    """
    cleaned = clean_input(user_input)
    input_vec = vectorizer.transform([cleaned])

    # Cosine similarity between user input and all movies
    scores = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # Get top_n indices (excluding perfect 1.0 if the exact movie is in dataset)
    top_indices = scores.argsort()[::-1][:top_n]

    results = df.iloc[top_indices][["Movie Name", "Storyline"]].copy()
    results["Similarity Score"] = scores[top_indices].round(4)
    results.insert(0, "Rank", range(1, len(results) + 1))
    results.reset_index(drop=True, inplace=True)

    return results
