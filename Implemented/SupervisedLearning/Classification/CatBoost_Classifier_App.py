# CATBOOST CLASSIFIER DEMO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from catboost import CatBoostClassifier

def catboost_page():
    st.sidebar.title("🐱 Supervised Learning (CatBoost Classifier)")
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
    iterations = st.sidebar.slider("Iterations", 50, 500, 200, step=50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
    depth = st.sidebar.slider("Tree Depth", 2, 10, 6)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate,
        depth=depth, verbose=0, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("🐱 CatBoost Classifier Analysis")

    with st.expander("❓ What is CatBoost?"):
        st.markdown("""
        **CatBoost** is a gradient boosting library by Yandex, particularly strong with categorical features.  

        ✅ Handles categorical data automatically  
        ✅ Requires little preprocessing  
        ✅ Great accuracy and less prone to overfitting  
        """)

    st.subheader("📊 Performance Metrics")
    metrics_df = pd.DataFrame({
        "Accuracy": [accuracy_score(y_test, y_pred)],
        "Precision": [precision_score(y_test, y_pred, average="weighted", zero_division=0)],
        "Recall": [recall_score(y_test, y_pred, average="weighted", zero_division=0)],
        "F1 Score": [f1_score(y_test, y_pred, average="weighted", zero_division=0)]
    }).round(3)
    st.dataframe(metrics_df)

    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names, ax=ax)
    st.pyplot(fig)

    with st.expander("📑 Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=class_names))

    st.subheader("⭐ Feature Importances")
    if X.shape[1] <= 30:
        importances = pd.Series(model.get_feature_importance(), index=X.columns)
        fig, ax = plt.subplots()
        importances.sort_values().plot(kind="barh", ax=ax, color="purple")
        st.pyplot(fig)
