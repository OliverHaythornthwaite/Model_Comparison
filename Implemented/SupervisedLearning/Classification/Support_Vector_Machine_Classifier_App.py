# SVM CLASSIFIER DEMO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def svm_page():
    st.sidebar.title("⚖️ Supervised Learning (SVM Classifier)")
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

    st.sidebar.subheader("⚙️ Model Parameters")
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("⚖️ Support Vector Machine Classifier Analysis")

    with st.expander("❓ What is SVM?"):
        st.markdown("""
        SVM finds the **optimal hyperplane** that separates classes with the **maximum margin**.  

        ✅ Effective in high-dimensional spaces  
        ✅ Works well with small/medium datasets  
        ⚠️ Can be slow on very large datasets  
        """)

    # Metrics
    st.subheader("📊 Performance Metrics")
    metrics_df = pd.DataFrame({
        "Accuracy": [accuracy_score(y_test, y_pred)],
        "Precision": [precision_score(y_test, y_pred, average="weighted", zero_division=0)],
        "Recall": [recall_score(y_test, y_pred, average="weighted", zero_division=0)],
        "F1 Score": [f1_score(y_test, y_pred, average="weighted", zero_division=0)]
    }).round(3)
    st.dataframe(metrics_df)

    # Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=ax)
    st.pyplot(fig)

    # Classification Report
    with st.expander("📑 Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=class_names))
