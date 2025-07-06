#UNSUPERVISED DIMENSIONALITY REDUCTION COMPARISON
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap  # Requires `pip install umap-learn`
from sklearn.preprocessing import StandardScaler

def dimensionality_reduction_explorer():
    @st.cache_data
    def load_dataset(name):
        if name == "Iris":
            data = load_iris()
            return data.data, data.target, data.target_names
        elif name == "Digits":
            data = load_digits()
            return data.data, data.target, None
        elif name == "Wine":
            data = load_wine()
            return data.data, data.target, data.target_names

    st.sidebar.title("Dimensionality Reduction Explorer")
    dataset_name = st.sidebar.selectbox("Choose Dataset", ["Iris", "Digits", "Wine"])
    method = st.sidebar.selectbox("Choose Method", ["PCA", "t-SNE", "UMAP"])
    perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30) if method == "t-SNE" else None
    n_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 50, 15) if method == "UMAP" else None
    min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 0.99, 0.1) if method == "UMAP" else None

    X, y, target_names = load_dataset(dataset_name)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.title("🔽 Dimensionality Reduction")

    with st.expander("❓ What is Dimensionality Reduction and Why Use It?"):
        st.markdown("""
        ### Understanding Dimensionality Reduction

        **Dimensionality Reduction** is a set of techniques used to reduce the number of input variables (features) in a dataset while preserving as much important information as possible.

        #### Why Reduce Dimensions?

        - Many real-world datasets have hundreds or thousands of features, which can:
          - Slow down learning algorithms
          - Cause overfitting
          - Make visualization and interpretation difficult
        - Reducing dimensions simplifies data, making it easier to analyze and visualize.

        #### Common Dimensionality Reduction Techniques

        - **Principal Component Analysis (PCA)**: Projects data onto orthogonal components that capture the most variance.
        - **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Non-linear technique for visualizing high-dimensional data in 2D or 3D.
        - **Uniform Manifold Approximation and Projection (UMAP)**: Preserves global and local structure for visualization.
        - **Autoencoders**: Neural networks that learn compressed representations.

        #### Benefits of Dimensionality Reduction

        - Improves model performance and speed by removing redundant/noisy features.
        - Enables visualization of complex high-dimensional data.
        - Helps uncover hidden structures or patterns.

        #### How to Interpret Results?

        - Principal components or reduced features represent combinations of original variables.
        - Visual clusters or patterns in reduced space reveal relationships.
        - Reconstruction error or explained variance indicates quality of reduction.

        ---

        This app applies dimensionality reduction algorithms to sample datasets to help you visualize and understand high-dimensional data in a simpler form.
        """)

    if method == "PCA":
        pca = PCA(n_components=2)
        X_embedded = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
        st.write(f"Explained variance by 2 components: **{explained_var:.2%}**")

    elif method == "t-SNE":
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_embedded = tsne.fit_transform(X_scaled)

    elif method == "UMAP":
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        X_embedded = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='Spectral', s=50, alpha=0.8)
    ax.set_title(f"{method} embedding of {dataset_name} dataset")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if target_names is not None:
        handles, _ = scatter.legend_elements()
        handles = list(handles)
        labels = list(target_names)

        if len(handles) == len(labels):
            legend1 = ax.legend(handles=handles, labels=labels, title="Classes")
            ax.add_artist(legend1)
        else:
            ax.legend(labels=labels, title="Classes")
    else:
        plt.colorbar(scatter, ax=ax, label="Class label")

    st.pyplot(fig)