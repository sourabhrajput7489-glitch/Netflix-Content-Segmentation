
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Netflix Content Segmentation",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "kmeans_model.pkl"

CLUSTER_NAMES = {
    0: "Comedy Movies",
    1: "Action & Cult Movies",
    2: "International & General TV Shows",
    3: "Documentary Movies",
    4: "Children & Family Movies",
    5: "Drama & Mixed-Genre Movies",
    6: "TV Action & Adventure",
    7: "Crime TV Shows",
    8: "Classic Movies"
}

# -------------------------------
# Load Dataset and Model
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    df = load_data()
    artifacts = load_model()
except (FileNotFoundError, OSError) as e:
    st.error(f"Required file not found: {e}")
    st.stop()

kmeans = artifacts["kmeans"]
scaler = artifacts["scaler"]
pca = artifacts["pca"]
feature_columns = artifacts["feature_columns"]

# -------------------------------
# Feature Engineering
# -------------------------------
data = df.copy()

data["duration_value"] = (
    data["duration"]
    .str.extract(r"(\d+)")[0]
    .astype(float)
)

data["type_encoded"] = data["type"].map({
    "Movie": 0,
    "TV Show": 1
})

data["primary_genre"] = (
    data["listed_in"]
    .str.split(",")
    .str[0]
    .str.strip()
)

features = data[
    ["release_year", "duration_value",
     "type_encoded", "primary_genre"]
].copy()

features = pd.get_dummies(
    features,
    columns=["primary_genre"],
    dtype=int
)

features = features.reindex(
    columns=feature_columns,
    fill_value=0
)

X_scaled = scaler.transform(features)

data["cluster"] = kmeans.predict(X_scaled)

X_pca = pca.transform(X_scaled)

data["PCA1"] = X_pca[:, 0]
data["PCA2"] = X_pca[:, 1]

data["segment"] = data["cluster"].map(CLUSTER_NAMES)

# -------------------------------
# Dashboard Header
# -------------------------------
st.title("🎬 Netflix Content Segmentation")

st.write(
    "Discover Netflix content groups using "
    "K-Means Clustering and Unsupervised Machine Learning."
)

st.divider()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📊 Dashboard Controls")

selected_clusters = st.sidebar.multiselect(
    "Select Clusters",
    options=sorted(data["cluster"].unique()),
    default=sorted(data["cluster"].unique()),
    format_func=lambda x: f"{x} - {CLUSTER_NAMES.get(x, 'Cluster')}"
)

filtered_data = data[
    data["cluster"].isin(selected_clusters)
].copy()

# -------------------------------
# KPI Metrics
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(filtered_data))

col2.metric(
    "Movies",
    (filtered_data["type"] == "Movie").sum()
)

col3.metric(
    "TV Shows",
    (filtered_data["type"] == "TV Show").sum()
)

col4.metric(
    "Active Clusters",
    filtered_data["cluster"].nunique()
)

st.divider()

# -------------------------------
# PCA Cluster Visualization
# -------------------------------
st.subheader("🔍 Netflix Cluster Visualization")

fig = px.scatter(
    filtered_data,
    x="PCA1",
    y="PCA2",
    color="segment",
    hover_data=[
        "title", "type", "release_year",
        "primary_genre", "rating"
    ],
    title="Netflix Content Segmentation Using PCA",
    opacity=0.65
)

fig.update_layout(
    height=550,
    legend_title_text="Content Segment"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Cluster Distribution
# -------------------------------
st.subheader("📈 Cluster Distribution")

cluster_counts = (
    filtered_data.groupby(
        ["cluster", "segment"]
    )
    .size()
    .reset_index(name="Total Titles")
)

fig2 = px.bar(
    cluster_counts,
    x="segment",
    y="Total Titles",
    color="segment",
    title="Number of Netflix Titles per Cluster"
)

fig2.update_layout(
    height=450,
    xaxis_tickangle=-35,
    showlegend=False
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Genre Analysis
# -------------------------------
st.subheader("🎭 Genre Analysis")

genre_counts = (
    filtered_data["primary_genre"]
    .value_counts()
    .head(12)
    .reset_index()
)

genre_counts.columns = ["Genre", "Count"]

fig3 = px.bar(
    genre_counts,
    x="Count",
    y="Genre",
    orientation="h",
    title="Top Netflix Genres"
)

fig3.update_layout(
    height=500,
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# Cluster Characteristics
# -------------------------------
st.subheader("📋 Cluster Characteristics")

cluster_summary = (
    filtered_data.groupby(["cluster", "segment"])
    .agg(
        total_titles=("title", "count"),
        avg_release_year=("release_year", "mean"),
        movies=("type", lambda x: (x == "Movie").sum()),
        tv_shows=("type", lambda x: (x == "TV Show").sum())
    )
    .reset_index()
)

st.dataframe(
    cluster_summary.round(2),
    use_container_width=True
)

# -------------------------------
# Netflix Content Explorer
# -------------------------------
st.subheader("🎥 Explore Netflix Titles")

search_title = st.text_input(
    "Search Netflix Title",
    placeholder="Enter movie or TV show name..."
)

explorer_data = filtered_data.copy()

if search_title:
    explorer_data = explorer_data[
        explorer_data["title"].str.contains(
            search_title,
            case=False,
            na=False,
            regex=False
        )
    ]

st.dataframe(
    explorer_data[
        [
            "title", "type", "release_year",
            "rating", "primary_genre",
            "country", "segment"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Machine Learning Internship Project | "
    "K-Means Clustering | Netflix Content Segmentation"
)
