#UNSUPERVISED ANOMOLY DETECTION COMPARISON
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def isolation_forest_page():
    st.title("🔽 Dimensionality Reduction")

    with st.expander("❓ What is Dimensionality Reduction and Why Use It?"):
        st.markdown("""
        ### Understanding Dimensionality Reduction

        **Dimensionality Reduction** is the process of reducing the number of random variables under consideration, by obtaining a set of principal variables.

        #### Why do we need Dimensionality Reduction?

        - Many datasets have a large number of features, which can lead to:
          - **Increased computational cost**
          - **Overfitting of machine learning models**
          - **Difficulty in visualization**
        - Dimensionality reduction helps to simplify data without losing essential information.

        #### Common Techniques

        - **Principal Component Analysis (PCA)**: Projects data into fewer dimensions by maximizing variance.
        - **t-SNE (t-distributed Stochastic Neighbor Embedding)**: Non-linear method that preserves local structure, good for visualization.
        - **UMAP (Uniform Manifold Approximation and Projection)**: Captures both local and global data structure, faster than t-SNE.
        - **Autoencoders**: Neural networks that learn compressed data representations.

        #### Benefits

        - Speeds up training of models by reducing feature space.
        - Improves model generalization by removing noise/redundant features.
        - Enables visualization of high-dimensional data in 2D or 3D.
        - Helps uncover hidden patterns and relationships.

        #### Interpreting Results

        - Reduced dimensions represent combinations of original features.
        - Visualization of reduced data can reveal natural groupings or clusters.
        - Explained variance (in PCA) indicates how much information is retained.

        ---

        This app demonstrates how dimensionality reduction techniques simplify complex datasets and help reveal underlying data structures.
        """)

    # Load data
    data = load_iris()
    X = data.data
    feature_names = data.feature_names
    target = data.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sidebar parameters
    contamination = st.sidebar.slider("Contamination (expected anomaly proportion)", 0.01, 0.2, 0.1, 0.01)

    # Train Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X_scaled)

    # Predict anomalies: -1 is anomaly, 1 is normal
    preds = clf.predict(X_scaled)
    df = pd.DataFrame(X, columns=feature_names)
    df['Anomaly'] = preds
    df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    # Show anomaly counts
    counts = df['Anomaly'].value_counts()
    st.write(f"Normal points: {counts.get('Normal', 0)}")
    st.write(f"Anomalies detected: {counts.get('Anomaly', 0)}")

    # Plot anomaly vs normal on first two features
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature_names[0], y=feature_names[1], hue='Anomaly', palette=['green', 'red'], ax=ax)
    ax.set_title("Isolation Forest Anomaly Detection")
    st.pyplot(fig)