# MLP CLASSIFIER DEMO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def mlp_classifier_page():
    st.sidebar.title("🧠 MLP Classifier")
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
    hidden_layer_sizes = st.sidebar.slider("Hidden Layer Size", 10, 200, 50)
    activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh", "logistic"])
    solver = st.sidebar.selectbox("Solver", ["adam", "sgd"])
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, stratify=y, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation=activation, solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("🧠 Multi-Layer Perceptron (MLP) Classifier Demo")

    with st.expander("❓ What is an MLP Classifier?"):
        st.markdown("""
        A Multi-Layer Perceptron (MLP) is a **feedforward neural network**.  
        - Composed of input, hidden, and output layers  
        - Learns non-linear decision boundaries  

        ✅ Can model complex patterns  
        ✅ Flexible number of layers and neurons  
        ⚠️ Sensitive to feature scaling and hyperparameters  
        """)

    # Metrics
    st.subheader("📊 Performance Metrics")
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
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names, ax=ax)
    st.pyplot(fig)

    with st.expander("📑 Classification Report"):
        st.text(classification_report(y_test, y_pred, target_names=class_names))
