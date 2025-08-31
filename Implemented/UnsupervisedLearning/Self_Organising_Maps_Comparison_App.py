#UNSUPERVISED SELF ORGANISING MAPS COMPARISON.
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def som_clustering_page():
    st.title("Unsupervised Learning: Self-Organizing Map (SOM) Clustering")

    with st.expander("❓ What is a Self-Organizing Map (SOM)?"):
        st.markdown("""
        ### Understanding Self-Organizing Maps (SOM)

        A **Self-Organizing Map (SOM)** is an unsupervised neural network technique used to visualize and cluster high-dimensional data by projecting it onto a lower-dimensional (usually 2D) grid.

        #### Core Idea

        - SOMs learn to **map** complex, high-dimensional data onto a 2D lattice of neurons.
        - Each neuron has a weight vector representing a prototype in the original feature space.
        - During training, neurons compete to best represent input samples; winning neurons and their neighbors adjust their weights to become more like the input.

        #### How It Works

        1. **Initialization**: The SOM grid is initialized with random weight vectors.
        2. **Competition**: For each input data point, find the neuron whose weight is closest to the input (Best Matching Unit - BMU).
        3. **Cooperation**: Neighboring neurons around the BMU are identified based on a neighborhood function.
        4. **Adaptation**: The BMU and its neighbors adjust their weights toward the input vector.
        5. Repeat for many iterations, gradually reducing the neighborhood size and learning rate.

        #### Why Use SOMs?

        - They perform **dimensionality reduction** preserving the topological properties of data.
        - Useful for **clustering**, **visualization**, and **pattern discovery**.
        - Help detect natural groupings in data without needing labels (unsupervised).

        #### Visualization

        The resulting 2D map shows how data points cluster and relate to each other in the original space, making it easier to interpret complex datasets.

        #### Applications

        - Market segmentation
        - Bioinformatics (e.g., gene expression patterns)
        - Image and signal processing
        - Exploratory data analysis

        ---

        In this app, we apply SOM to the Iris dataset to visualize and cluster the flowers based on their features, showing how the algorithm groups similar samples together on the grid.
        """)

    # Load Iris data
    data = load_iris()
    X = data.data
    target = data.target
    target_names = data.target_names

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SOM parameters in sidebar
    som_x = st.sidebar.slider("SOM grid width (x)", 3, 10, 6)
    som_y = st.sidebar.slider("SOM grid height (y)", 3, 10, 6)
    sigma = st.sidebar.slider("Sigma (Neighborhood radius)", 0.1, 2.0, 1.0)
    learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.5)

    # Initialize and train SOM
    som = MiniSom(som_x, som_y, X_scaled.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=42)
    som.train_random(X_scaled, 500)

    # Map each sample to its winning neuron
    win_map = np.array([som.winner(x) for x in X_scaled])

    # Create DataFrame of mapped positions and original labels
    df_som = pd.DataFrame(win_map, columns=["X", "Y"])
    df_som['Label'] = target

    # Plot SOM grid with samples
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot neurons grid
    for i in range(som_x):
        for j in range(som_y):
            ax.plot(i + 0.5, j + 0.5, 'o', markerfacecolor='None', markeredgecolor='gray', markersize=20)

    # Plot data points on SOM grid colored by true class
    colors = ['r', 'g', 'b']
    for c, label_name in enumerate(target_names):
        cluster_points = df_som[df_som['Label'] == c]
        ax.scatter(
            cluster_points['X'] + 0.5 + (np.random.rand(len(cluster_points)) - 0.5)*0.6,
            cluster_points['Y'] + 0.5 + (np.random.rand(len(cluster_points)) - 0.5)*0.6,
            c=colors[c], label=label_name, alpha=0.7, edgecolors='k'
        )

    ax.set_xticks(np.arange(som_x + 1))
    ax.set_yticks(np.arange(som_y + 1))
    ax.set_xlim(0, som_x)
    ax.set_ylim(0, som_y)
    ax.grid(True)
    ax.set_title("Self-Organizing Map (SOM) Clustering of Iris Data")
    ax.legend(title="True Class")

    st.pyplot(fig)