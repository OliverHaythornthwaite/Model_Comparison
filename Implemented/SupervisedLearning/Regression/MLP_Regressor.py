# MLP REGRESSOR DEMO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mlp_regressor_page():
    st.sidebar.title("🧠 MLP Regressor")
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Diabetes", "California Housing"))

    @st.cache_data
    def load_dataset(name):
        if name == "Diabetes":
            data = load_diabetes()
            return pd.DataFrame(data.data, columns=data.feature_names), data.target
        elif name == "California Housing":
            data = fetch_california_housing()
            return pd.DataFrame(data.data, columns=data.feature_names), data.target

    X, y = load_dataset(dataset_name)

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

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes,), activation=activation, solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("🧠 Multi-Layer Perceptron (MLP) Regressor Demo")

    with st.expander("❓ What is an MLP Regressor?"):
        st.markdown("""
        A Multi-Layer Perceptron (MLP) Regressor is a **feedforward neural network for regression**.  
        - Can learn complex non-linear relationships between features and target  
        - Requires feature scaling for optimal performance  

        ✅ Models non-linear data  
        ✅ Flexible architecture  
        ⚠️ Sensitive to hyperparameters and data scaling  
        """)

    # Metrics
    st.subheader("📈 Performance Metrics")
    metrics = {
        "R² Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    st.dataframe(pd.DataFrame([metrics]).round(3))

    # Predictions vs Actual
    st.subheader("📉 Predictions vs Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)
