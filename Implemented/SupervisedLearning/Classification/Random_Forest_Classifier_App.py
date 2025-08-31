# RANDOM FOREST CLASSIFIER DEMO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def random_forest_page():
    st.sidebar.title("🌲 Supervised Learning (Random Forest Classifier)")
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Digits"))

    @st.cache_data
    def load_dataset(name):
        if name == "Iris":
            data = load_iris()
            return pd.DataFrame(data.data, columns=data.feature_names), data.target, data.target_names
        elif name == "Wine":
            data = load_wine()
            return pd.DataFrame(data.data, columns=data.feature_names), data.target, data.target_names
        elif name == "Digits":
            data = load_digits()
            return pd.DataFrame(data.data), data.target, np.arange(10)

    X, y, class_names = load_dataset(dataset_name)

    # Sidebar hyperparameters
    st.sidebar.subheader("⚙️ Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 10, 300, 100, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 30, 5)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("🌲 Random Forest Classifier Analysis")

    with st.expander("❓ What is a Random Forest?"):
        st.markdown("""
        ### Understanding Random Forests  
        A **Random Forest** is an **ensemble method** that builds multiple decision trees and averages their predictions.

        #### Why Random Forests?
        - Reduces overfitting compared to a single decision tree.  
        - Provides feature importance for interpretation.  
        - Works well on many datasets out of the box.  
        """)

    # Metrics
    metrics_df = pd.DataFrame({
        "Accuracy": [accuracy_score(y_test, y_pred)],
        "Precision": [precision_score(y_test, y_pred, average="weighted", zero_division=0)],
        "Recall": [recall_score(y_test, y_pred, average="weighted", zero_division=0)],
        "F1 Score": [f1_score(y_test, y_pred, average="weighted", zero_division=0)]
    }).round(3)
    st.subheader("📊 Performance Metrics")
    st.dataframe(metrics_df)

    # Confusion matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # Classification report
    with st.expander("📑 Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=class_names))

    # Feature importance
    st.subheader("⭐ Feature Importances")
    if X.shape[1] <= 30:  # Avoid plotting too many
        importances = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        importances.sort_values().plot(kind="barh", ax=ax, color="forestgreen")
        st.pyplot(fig)
