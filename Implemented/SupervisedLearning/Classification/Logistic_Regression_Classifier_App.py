# LOGISTIC REGRESSION CLASSIFIER DEMO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def logistic_regression_page():
    st.sidebar.title("📊 Logistic Regression Classifier")
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
            return pd.DataFrame(data.data), data.target, [str(i) for i in range(10)]

    X, y, class_names = load_dataset(dataset_name)

    # Sidebar parameters
    st.sidebar.subheader("⚙️ Model Parameters")
    max_iter = st.sidebar.slider("Max Iterations", 50, 500, 200)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=max_iter, solver="lbfgs", multi_class="auto")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("📊 Logistic Regression Classifier Demo")

    # Explanatory section
    with st.expander("❓ What is Logistic Regression?"):
        st.markdown("""
        Logistic Regression is a **linear classifier** used to predict categorical outcomes.  
        It estimates probabilities using the logistic (sigmoid) function and assigns classes based on thresholds.  
        
        ✅ Works well for linearly separable problems  
        ✅ Simple and interpretable  
        ⚠️ Limited for complex, non-linear decision boundaries
        """)

    # Metrics
    st.subheader("📈 Performance Metrics")
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }
    st.dataframe(pd.DataFrame([metrics]).round(3))

    # Confusion matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    st.pyplot(fig)

    with st.expander("📑 Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=class_names))
