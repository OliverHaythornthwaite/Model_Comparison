# DECISION TREE CLASSIFIER COMPARISON
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

def decision_tree_page():
    st.sidebar.title("🌳 Supervised Learning (Decision Tree Classifier)")
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

    # Sidebar model hyperparameters
    st.sidebar.subheader("⚙️ Model Parameters")
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 3)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
    test_size = st.sidebar.slider("Test Size (for validation)", 0.1, 0.5, 0.3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("🌳 Decision Tree Classifier Analysis")

    with st.expander("❓ What is a Decision Tree Classifier?"):
        st.markdown("""
        ### Understanding Decision Trees
        A **Decision Tree** is a supervised learning model used for **classification** and **regression**.

        #### How It Works
        - The algorithm splits the dataset into branches based on feature values.
        - Each internal node represents a decision rule on a feature.
        - Each leaf node corresponds to a predicted class.

        #### Advantages
        - Easy to interpret and visualize.
        - Handles both numerical and categorical data.
        - No need for feature scaling.

        #### Limitations
        - Can easily **overfit** if not pruned or limited by depth.
        - Small changes in data may produce very different trees.

        This demo shows how changing depth and criterion affects performance.
        """)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    st.subheader("📊 Performance Metrics")
    metrics_df = pd.DataFrame({
        "Accuracy": [acc],
        "Precision": [prec],
        "Recall": [rec],
        "F1 Score": [f1]
    }).round(3)
    st.dataframe(metrics_df)

    # Confusion matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # Textual classification report
    with st.expander("📑 Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=class_names))

    # Visualize the decision tree
    st.subheader("🌲 Decision Tree Structure")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=class_names.astype(str),
              filled=True, rounded=True, fontsize=8, ax=ax)
    st.pyplot(fig)
