#UNSUPERVISED CLUSTERING COMPARISON
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_digits, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def clustering_comparison_page():
    @st.cache_data
    def load_dataset(name):
        if name == "Iris":
            data = load_iris()
            return pd.DataFrame(data.data, columns=data.feature_names), data.target
        elif name == "Digits":
            data = load_digits()
            return pd.DataFrame(data.data), data.target
        elif name == "Synthetic":
            X, y = make_blobs(n_samples=300, centers=4, n_features=4, random_state=42)
            return pd.DataFrame(X), y

    st.sidebar.title("🧪 Unsupervised Learning (Clustering)")
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Digits", "Synthetic"))

    with st.sidebar.expander("📘 Metric Descriptions"):
        st.markdown("""
        **Silhouette Score**: Measures how similar an object is to its own cluster vs other clusters. Higher is better.  
        **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.  
        **Davies-Bouldin Index**: Measures intra-cluster similarity and inter-cluster difference. Lower is better.  
        """)

    X, y_true = load_dataset(dataset_name)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "KMeans (k=3)": KMeans(n_clusters=3, random_state=42),
        "Agglomerative Clustering (k=3)": AgglomerativeClustering(n_clusters=3),
        "DBSCAN": DBSCAN(eps=0.7, min_samples=5)
    }

    selected_models = st.sidebar.multiselect("Choose Clustering Models", list(models.keys()), default=list(models.keys()))

    st.title("🔍 Clustering Analysis")

    with st.expander("❓ What is Clustering and Why Use It?"):
        st.markdown("""
        ### Understanding Clustering

        **Clustering** is an **unsupervised learning** technique used to group data points into clusters based on their similarity, without using predefined labels.

        #### What is Clustering?

        - It finds natural groupings or patterns in data by grouping similar points together.
        - Each cluster contains data points that are more similar to each other than to those in other clusters.
        - It helps uncover the underlying structure of data.

        #### Common Clustering Algorithms

        - **K-Means**: Partitions data into *k* clusters by minimizing within-cluster variance.
        - **Hierarchical Clustering**: Builds clusters step-by-step using a tree-like structure.
        - **DBSCAN**: Groups points based on density; good for detecting arbitrary-shaped clusters and noise.
        - **Self-Organizing Maps (SOM)**: Neural network that maps high-dimensional data onto a 2D grid preserving topology.

        #### Why Use Clustering?

        - To discover groups or segments in data (e.g., customer segmentation).
        - To reduce data complexity by summarizing patterns.
        - To detect anomalies or outliers.
        - To visualize high-dimensional data in a simplified form.

        #### How to Interpret Clustering Results?

        - Examine cluster centroids or representative points.
        - Use visualization tools to see cluster separation.
        - Combine domain knowledge to understand what each cluster represents.

        ---

        This app applies clustering algorithms to example datasets to help you explore and visualize how data naturally groups together.
        """)

    results = []

    def evaluate_model(name, model):
        labels = model.fit_predict(X_scaled)

        if len(set(labels)) <= 1:
            st.warning(f"{name}: Clustering failed (only one cluster). Skipping metrics.")
            return

        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)

        results.append({
            "Model": name,
            "Silhouette": sil,
            "Calinski-Harabasz": ch,
            "Davies-Bouldin": db
        })

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50)
        ax.set_title(f"{name} - PCA Projection")
        st.pyplot(fig)

    if selected_models:
        for model_name in selected_models:
            st.subheader(f"📌 Evaluating: {model_name}")
            model = models[model_name]
            evaluate_model(model_name, model)

        if results:
            df_results = pd.DataFrame(results).set_index("Model").round(3)
            st.subheader("📊 Clustering Performance Metrics")
            st.dataframe(df_results)

            st.subheader("📈 Compare Metric Across Models")
            metric_to_plot = st.selectbox("Choose metric", df_results.columns)
            fig, ax = plt.subplots()
            df_results[metric_to_plot].plot(kind="bar", ax=ax, color="cornflowerblue")
            ax.set_ylabel(metric_to_plot)
            st.pyplot(fig)
    else:
        st.warning("Please select at least one clustering model to evaluate.")