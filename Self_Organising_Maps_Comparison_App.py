import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

st.title("Unsupervised Learning: Self-Organizing Map (SOM) Clustering")

# Load Iris data
data = load_iris()
X = data.data
target = data.target
target_names = data.target_names
feature_names = data.feature_names

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
    ax.scatter(cluster_points['X'] + 0.5 + (np.random.rand(len(cluster_points)) - 0.5)*0.6,
               cluster_points['Y'] + 0.5 + (np.random.rand(len(cluster_points)) - 0.5)*0.6,
               c=colors[c], label=label_name, alpha=0.7, edgecolors='k')

ax.set_xticks(np.arange(som_x + 1))
ax.set_yticks(np.arange(som_y + 1))
ax.set_xlim(0, som_x)
ax.set_ylim(0, som_y)
ax.grid(True)
ax.set_title("Self-Organizing Map (SOM) Clustering of Iris Data")
ax.legend(title="True Class")

st.pyplot(fig)