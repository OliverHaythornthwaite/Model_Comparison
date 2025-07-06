#SUPERVISED CLASSIFIER COMPARISON.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    log_loss
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import label_binarize

def classification_comparison_page():
    # Sidebar: Dataset selection
    st.sidebar.title("Dataset and Model Selection")

    dataset_name = st.sidebar.selectbox(
        "Choose Classification Dataset",
        ("Iris", "Wine", "Breast Cancer", "Digits")
    )

    # Sidebar: Performance Metric Descriptions
    with st.sidebar.expander("📘 Performance Metric Descriptions", expanded=False):
        st.markdown("""
        **Accuracy**: The proportion of correct predictions over total predictions.  
        **Precision**: The ratio of true positives to all predicted positives (TP / (TP + FP)).  
        **Recall (Sensitivity)**: The ratio of true positives to all actual positives (TP / (TP + FN)).  
        **F1 Score**: Harmonic mean of precision and recall; balances false positives and false negatives.  
        **ROC AUC**: Measures model's ability to distinguish between classes. Higher is better (closer to 1).  
        **Log Loss**: Measures uncertainty in predictions. Lower is better.  
        """)

    @st.cache_data
    def load_selected_data(name):
        if name == "Iris":
            data = load_iris()
        elif name == "Wine":
            data = load_wine()
        elif name == "Breast Cancer":
            data = load_breast_cancer()
        elif name == "Digits":
            data = load_digits()
        else:
            raise ValueError("Unsupported dataset")
        return data.data, data.target, data.target_names

    X, y, class_names = load_selected_data(dataset_name)

    st.title("🧠 Classifier Model Comparison")

    with st.expander("❓ What is Classification and Why Compare Models?"):
        st.markdown("""
        ### Understanding Classification and Model Comparison

        **Classification** is a core task in supervised machine learning where the goal is to assign input data to one of several discrete categories or classes.  
        For example, determining if an email is spam or not, or recognizing handwritten digits.

        #### What Is Classification?

        - It learns a function that maps input features to discrete labels.
        - The model predicts the class label for new, unseen data based on patterns learned during training.

        #### Why Compare Multiple Classification Models?

        - Different classifiers have different strengths, weaknesses, and assumptions.
        - Some models (like Logistic Regression) are simple and interpretable.
        - Others (like Random Forest or Gradient Boosting) often yield higher accuracy but are more complex.
        - Comparing models helps find the best fit for your specific data and task.

        #### Common Classifiers in This Comparison

        - **Logistic Regression**: A linear model for binary classification.
        - **K-Nearest Neighbors (KNN)**: Classifies based on the majority class of nearest neighbors.
        - **Decision Trees**: Models decisions as a tree of conditions.
        - **Random Forest**: Ensemble of decision trees improving robustness and accuracy.
        - **Support Vector Machines (SVM)**: Finds the optimal separating hyperplane.
        - **Gradient Boosting (XGBoost, LightGBM, CatBoost)**: Powerful ensemble methods using boosting techniques.

        #### Performance Metrics

        - **Accuracy**: Percentage of correctly classified instances.
        - **Precision and Recall**: Metrics that balance false positives and false negatives.
        - **F1-Score**: Harmonic mean of precision and recall.
        - **ROC-AUC**: Measures the ability to distinguish between classes.

        #### Why This Matters

        Different problems and datasets may favor different classifiers.  
        By comparing multiple models on the same data using consistent metrics, you can make informed decisions that balance accuracy, interpretability, and computational cost.

        ---

        This app trains several classification algorithms on sample datasets, evaluates them using key metrics, and visualizes their performance to help you explore and understand classifier behavior.
        """)
    st.markdown(f"### Dataset: {dataset_name}")
    st.write("Number of samples:", X.shape[0])
    st.write("Number of features:", X.shape[1])
    st.write("Number of classes:", len(class_names))

    y_binarized = label_binarize(y, classes=np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    st.sidebar.title("Select Classification Models to Compare")
    selected_models = st.sidebar.multiselect("Choose Models", list(models.keys()), default=list(models.keys()))

    results = []

    def evaluate_model(name, model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.unique(y))

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        logloss = log_loss(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc,
            "Log Loss": logloss
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve
        fig, ax = plt.subplots()
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"Class {class_names[i]}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"{name} - ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        # Precision-Recall Curve
        fig, ax = plt.subplots()
        for i in range(len(class_names)):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
            ax.plot(recall, precision, label=f"Class {class_names[i]}")
        ax.set_title(f"{name} - Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)

    if selected_models:
        for model_name in selected_models:
            st.subheader(f"🧠 Evaluating: {model_name}")
            model = models[model_name]
            evaluate_model(model_name, model)

        metrics_df = pd.DataFrame(results).set_index("Model").round(4)
        st.subheader("📈 Model Performance Metrics")
        st.dataframe(metrics_df)

        st.subheader("📊 Metric Comparison Chart")
        selected_metric = st.selectbox("Select metric to plot", metrics_df.columns)

        fig, ax = plt.subplots()
        metrics_df[selected_metric].plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(selected_metric)
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        st.pyplot(fig)
    else:
        st.warning("Please select at least one model to evaluate.")