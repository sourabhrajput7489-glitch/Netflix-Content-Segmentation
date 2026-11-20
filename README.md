# Netflix Content Segmentation

Machine Learning Internship — Task 4 (Medium)

## Project Overview
This project groups 8,790 Netflix movies and TV shows into meaningful content segments using unsupervised machine learning.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- K-Means Clustering
- StandardScaler
- PCA
- Matplotlib and Seaborn
- Streamlit and Plotly

## Workflow
1. Load and explore Netflix data.
2. Extract duration and primary genre features.
3. Encode categorical features and scale numerical data.
4. Evaluate cluster counts using the Elbow Method and Silhouette Score.
5. Train K-Means with 9 clusters.
6. Visualize clusters using PCA.
7. Analyze cluster characteristics.
8. Build an interactive Streamlit dashboard.

## Model Results
- Dataset: 8,790 Netflix titles
- Algorithm: K-Means
- Selected clusters: 9
- Silhouette Score: approximately 0.3282

## Cluster Segments
0. Comedy Movies
1. Action & Cult Movies
2. International & General TV Shows
3. Documentary Movies
4. Children & Family Movies
5. Drama & Mixed-Genre Movies
6. TV Action & Adventure
7. Crime TV Shows
8. Classic Movies

## Run the Application

Install dependencies:

    pip install -r requirements.txt

Launch Streamlit:

    python -m streamlit run app.py

## Limitations
Clusters are influenced by encoded genre and content type features. The results describe content metadata patterns rather than audience preferences.

## Project Type
Unsupervised Machine Learning / Content Segmentation
