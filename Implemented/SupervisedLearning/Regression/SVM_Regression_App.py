# SUPPORT VECTOR REGRESSOR DEMO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def svr_regression_page():
    st.sidebar.title("📐 Support Vector Regression (SVR)")
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

    # Sidebar params
    st.sidebar.subheader("⚙️ Model Parameters")
    kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"))
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    # Scaling is important for SVR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("📐 Support Vector Regression (SVR) Demo")

    with st.expander("❓ What is Support Vector Regression?"):
        st.markdown("""
        SVR is based on **Support Vector Machines (SVMs)**.  
        - Instead of finding a classification boundary, it fits a regression function within a tolerance (`epsilon`).  

        ✅ Effective in high-dimensional spaces  
        ✅ Supports non-linear regression with kernels  
        ⚠️ Can be slow on very large datasets  
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
