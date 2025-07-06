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

def regression_comparison_page():
    @st.cache_data
    def load_dataset(name):
        if name == "Diabetes":
            data = load_diabetes()
            return pd.DataFrame(data.data, columns=data.feature_names), data.target
        elif name == "Synthetic":
            X, y = make_regression(n_samples=300, n_features=5, noise=20, random_state=42)
            return pd.DataFrame(X), y

    st.sidebar.title("Regression Model Comparison")
    dataset_name = st.sidebar.selectbox("Choose Dataset", ("Diabetes", "Synthetic"))

    with st.sidebar.expander("📘 Performance Metric Descriptions"):
        st.markdown("""
        **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values.  
        **MSE (Mean Squared Error)**: Average of squared differences. Penalizes large errors.  
        **RMSE (Root Mean Squared Error)**: Square root of MSE. Same units as target variable.  
        **R² (R-squared)**: Proportion of variance explained by the model. 1 = perfect prediction.  
        """)

    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(verbosity=0),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    selected_models = st.sidebar.multiselect("Choose Models", list(models.keys()), default=list(models.keys()))

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

        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, alpha=0.7)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_title(f"{name} - True vs Predicted")
        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predicted Values")
        st.pyplot(fig1)

        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots()
        ax2.hist(residuals, bins=30, color='orange', edgecolor='black')
        ax2.set_title(f"{name} - Residuals Histogram")
        ax2.set_xlabel("Residual")
        st.pyplot(fig2)

    st.title("📊 Regression Model Comparison")

    with st.expander("❓ What is Regression and Why Compare Models?"):
        st.markdown("""
        ### Understanding Regression and Model Comparison

        **Regression** is a fundamental supervised learning technique used to predict continuous outcomes based on input features.  
        For example, predicting house prices, sales amounts, or patient blood sugar levels.

        #### What Is Regression?

        - It models the relationship between one or more independent variables (features) and a continuous dependent variable (target).
        - The goal is to find a function that best fits the data and generalizes well to unseen samples.

        #### Why Compare Multiple Regression Models?

        - Different regression algorithms have different strengths, assumptions, and biases.
        - Some models handle non-linear relationships better (e.g., Random Forest, XGBoost).
        - Others offer interpretability and simplicity (e.g., Linear Regression).
        - Comparing models helps identify which works best for your data and problem.

        #### Common Regression Models in This Comparison

        - **Linear Regression**: Assumes a linear relationship between features and target.
        - **Ridge and Lasso Regression**: Linear models with regularization to prevent overfitting.
        - **Random Forest**: Ensemble of decision trees; good at capturing non-linearities.
        - **XGBoost, LightGBM, CatBoost**: Advanced gradient boosting methods, often providing state-of-the-art performance.

        #### Performance Metrics

        - **MAE (Mean Absolute Error)**: Average magnitude of errors, easy to interpret.
        - **MSE (Mean Squared Error)**: Penalizes larger errors more heavily.
        - **RMSE (Root Mean Squared Error)**: Square root of MSE; same units as target.
        - **R² (R-squared)**: Percentage of variance explained; closer to 1 means better fit.

        #### Why This Matters

        Comparing these models side-by-side on the same dataset allows you to understand trade-offs between accuracy, complexity, and interpretability.  
        It also highlights how model choice impacts predictive performance and helps you make informed decisions in real-world scenarios.

        ---

        This app demonstrates these concepts by training multiple regression models on sample datasets, evaluating them with key metrics, and visualizing their performance and predictions.
        """)


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