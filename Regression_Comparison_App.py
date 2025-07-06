#SUPERVISED REGRESSION COMPARISON
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
@st.cache_data
def load_dataset(name):
    if name == "Diabetes":
        data = load_diabetes()
        return pd.DataFrame(data.data, columns=data.feature_names), data.target
    elif name == "Synthetic":
        X, y = make_regression(n_samples=300, n_features=5, noise=20, random_state=42)
        return pd.DataFrame(X), y

# Sidebar: dataset and model selection
st.sidebar.title("Regression Model Comparison")
dataset_name = st.sidebar.selectbox("Choose Dataset", ("Diabetes", "Synthetic"))

with st.sidebar.expander("📘 Performance Metric Descriptions"):
    st.markdown("""
    **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values.  
    **MSE (Mean Squared Error)**: Average of squared differences. Penalizes large errors.  
    **RMSE (Root Mean Squared Error)**: Square root of MSE. Same units as target variable.  
    **R² (R-squared)**: Proportion of variance explained by the model. 1 = perfect prediction.  
    """)

# Load selected dataset
X, y = load_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(verbosity=0),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0)
}

# Model selection
selected_models = st.sidebar.multiselect("Choose Models", list(models.keys()), default=list(models.keys()))

# Evaluate models
results = []

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    })

    # Scatter: True vs Predicted
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.7)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_title(f"{name} - True vs Predicted")
    ax1.set_xlabel("True Values")
    ax1.set_ylabel("Predicted Values")
    st.pyplot(fig1)

    # Residual plot
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    ax2.hist(residuals, bins=30, color='orange', edgecolor='black')
    ax2.set_title(f"{name} - Residuals Histogram")
    ax2.set_xlabel("Residual")
    st.pyplot(fig2)

# App title
st.title("📊 Regression Model Comparison")

if selected_models:
    for name in selected_models:
        st.subheader(f"🔍 Evaluating: {name}")
        evaluate_model(name, models[name])

    df_results = pd.DataFrame(results).set_index("Model").round(3)
    st.subheader("📈 Model Performance Metrics")
    st.dataframe(df_results)

    st.subheader("📉 Metric Comparison")
    selected_metric = st.selectbox("Select metric to compare", df_results.columns)
    fig, ax = plt.subplots()
    df_results[selected_metric].plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel(selected_metric)
    ax.set_title(f"{selected_metric} by Model")
    st.pyplot(fig)
else:
    st.warning("Please select at least one model to evaluate.")