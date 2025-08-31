# LASSO REGRESSION DEMO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def lasso_regression_page():
    st.sidebar.title("✂️ Lasso Regression")
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
    alpha = st.sidebar.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("✂️ Lasso Regression Demo")

    with st.expander("❓ What is Lasso Regression?"):
        st.markdown("""
        Lasso Regression is a type of **linear regression with L1 regularization**.  
        It penalizes the absolute values of coefficients, shrinking some to **zero**, effectively performing **feature selection**.  

        ✅ Useful for sparse models  
        ✅ Can automatically drop irrelevant features  
        ⚠️ May underperform when many features are truly relevant  
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
