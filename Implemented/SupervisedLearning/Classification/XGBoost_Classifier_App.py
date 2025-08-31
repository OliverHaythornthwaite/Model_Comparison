# XGBOOST CLASSIFIER DEMO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from xgboost import XGBClassifier

def xgboost_page():
    st.sidebar.title("⚡ Supervised Learning (XGBoost Classifier)")
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
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100, step=50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, use_label_encoder=False, eval_metric="mlogloss", random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("⚡ XGBoost Classifier Analysis")

    with st.expander("❓ What is XGBoost?"):
        st.markdown("""
        **XGBoost** stands for **Extreme Gradient Boosting**.  
        It’s a highly optimized implementation of gradient boosting, famous for winning many Kaggle competitions.  

        ✅ Handles missing values  
        ✅ Efficient & scalable  
        ✅ Supports regularization to prevent overfitting  
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    st.pyplot(fig)

    # Report
    with st.expander("📑 Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=class_names))

    # Feature Importance
    st.subheader("⭐ Feature Importances")
    if X.shape[1] <= 30:
        importances = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        importances.sort_values().plot(kind="barh", ax=ax, color="dodgerblue")
        st.pyplot(fig)
