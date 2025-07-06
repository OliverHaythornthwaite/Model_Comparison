#UNSUPERVISED ANOMOLY DETECTION COMPARISON
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Unsupervised Anomaly Detection with Isolation Forest")

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
st.write(f"Normal points: {counts.get('Normal',0)}")
st.write(f"Anomalies detected: {counts.get('Anomaly',0)}")

# Plot anomaly vs normal on first two features
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df, x=feature_names[0], y=feature_names[1], hue='Anomaly', palette=['green', 'red'], ax=ax)
ax.set_title("Isolation Forest Anomaly Detection")
st.pyplot(fig)