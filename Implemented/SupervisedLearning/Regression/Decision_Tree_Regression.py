# DECISION TREE REGRESSOR DEMO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def decision_tree_regression_page():
    st.sidebar.title("🌳 Decision Tree Regression")
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
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.title("🌳 Decision Tree Regressor Demo")

    with st.expander("❓ What is Decision Tree Regression?"):
        st.markdown("""
        Decision Trees split data into regions using feature thresholds.  
        Each leaf node predicts a constant value (average of training samples in that leaf).  

        ✅ Easy to interpret and visualize  
        ✅ Captures non-linear relationships  
        ⚠️ Prone to overfitting without depth control  
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
