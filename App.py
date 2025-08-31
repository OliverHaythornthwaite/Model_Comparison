import streamlit as st

# ---- Supervised Learning (Classification) ----
from Implemented.SupervisedLearning.Classification.Classifier_Comparison_App import classification_comparison_page
from Implemented.SupervisedLearning.Classification.Decision_Tree_Classifier_App import decision_tree_page
from Implemented.SupervisedLearning.Classification.Random_Forest_Classifier_App import random_forest_page
from Implemented.SupervisedLearning.Classification.Gradient_Boosting_Classifier_App import gradient_boosting_page
from Implemented.SupervisedLearning.Classification.XGBoost_Classifier_App import xgboost_page
from Implemented.SupervisedLearning.Classification.LightGM_Classifier_App import lightgbm_page
from Implemented.SupervisedLearning.Classification.CatBoost_Classifier_App import catboost_page
from Implemented.SupervisedLearning.Classification.Naive_Bayes_Classifier_App import naive_bayes_page
from Implemented.SupervisedLearning.Classification.KNN_Classifier_App import knn_page
from Implemented.SupervisedLearning.Classification.Support_Vector_Machine_Classifier_App import svm_page
from Implemented.SupervisedLearning.Classification.Logistic_Regression_Classifier_App import logistic_regression_page
from Implemented.SupervisedLearning.Classification.MLP_Classifier_App import mlp_classifier_page

# ---- Supervised Learning (Regression) ----
from Implemented.SupervisedLearning.Regression.Regression_Comparison_App import regression_comparison_page
from Implemented.SupervisedLearning.Regression.Ridge_Regression_App import ridge_regression_page
from Implemented.SupervisedLearning.Regression.Lasso_Regression_App import lasso_regression_page
from Implemented.SupervisedLearning.Regression.ElasticNet_Regression_App import elasticnet_regression_page
from Implemented.SupervisedLearning.Regression.Decision_Tree_Regression import decision_tree_regression_page
from Implemented.SupervisedLearning.Regression.RandomForest_Regression_App import random_forest_regression_page
from Implemented.SupervisedLearning.Regression.Gradient_Boosting_Regressor_App import gradient_boosting_regression_page
from Implemented.SupervisedLearning.Regression.XGBoost_Regressor_App import xgboost_regression_page
from Implemented.SupervisedLearning.Regression.SVM_Regression_App import svr_regression_page
from Implemented.SupervisedLearning.Regression.MLP_Regressor import mlp_regressor_page

# ---- Unsupervised Learning ----
from Implemented.UnsupervisedLearning.Clustering_Comparison_App import clustering_comparison_page
from Implemented.UnsupervisedLearning.Dimensionality_Reduction_Comparison_App import dimensionality_reduction_explorer
from Implemented.UnsupervisedLearning.Self_Organising_Maps_Comparison_App import som_clustering_page

# ---- Reinforcement Learning ----
from Implemented.ReinforcementLearning.Basic_Q_Learning_Comparison_App import q_learning_frozenlake_page
from Implemented.ReinforcementLearning.Multi_Armed_Bandit_Comparison_App import multi_armed_bandit_page
from Implemented.ReinforcementLearning.Simple_Policy_Gradient_App import policy_gradient_cartpole_page
from Implemented.ReinforcementLearning.Swarming_Behaviour_App import swarming_behavior_page

# ---- Anomaly Detection ----
from Implemented.AnomalyDetection.Anomoly_Detection_Comparison_App import isolation_forest_page

# ---- Sensor Fusion ----
from Implemented.SensorFusion.Complementary_Filter_Sensor_Fusion_App import complementary_filter_sensor_fusion_page
from Implemented.SensorFusion.Kalman_Filter_Sensor_Fusion_App import kalman_filter_sensor_fusion_page
from Implemented.SensorFusion.Particle_filter_Sensor_Fuision_App import particle_filter_sensor_fusion_page

# ---- Emergent Behaviour ----
from Implemented.EmergentBehaviour.Flocking_Simulation_App import boids_flocking_page
from Implemented.EmergentBehaviour.Game_Of_Life import game_of_life_page
from Implemented.EmergentBehaviour.Swarm_Consensus import particle_swarming_page
from Implemented.EmergentBehaviour.Leader_Follower_Control import leader_follower_page
from Implemented.EmergentBehaviour.Goal_Based_Attractor import attractor_control_page
from Implemented.EmergentBehaviour.Parameter_Modulation import parameter_modulation_page

# ---- DO-178C AI Certification ----
from Implemented.D0178C.D0178C import do178c_ai_certification_page

# ---- Streamlit Page Config ----
st.set_page_config(page_title="ML Model Comparison Suite", layout="wide")

# ---- Categorize the pages for sidebar ----
categories = {
    "Supervised Learning (Classification)": [
        {"title": "Classification Model Comparison", "function": classification_comparison_page},
        {"title": "Decision Tree Classifier", "function": decision_tree_page},
        {"title": "Random Forest Classifier", "function": random_forest_page},
        {"title": "Gradient Boosting Classifier", "function": gradient_boosting_page},
        {"title": "XGBoost Classifier", "function": xgboost_page},
        {"title": "LightGBM Classifier", "function": lightgbm_page},
        {"title": "CatBoost Classifier", "function": catboost_page},
        {"title": "Naive Bayes Classifier", "function": naive_bayes_page},
        {"title": "K-Nearest Neighbors Classifier", "function": knn_page},
        {"title": "Support Vector Machine Classifier", "function": svm_page},
        {"title": "Logistic Regression Classifier", "function": logistic_regression_page},
        {"title": "Neural Net Classifier (MLP)", "function": mlp_classifier_page}
    ],
    "Supervised Learning (Regression)": [
        {"title": "Regression Model Comparison", "function": regression_comparison_page},
        {"title": "Ridge Regression", "function": ridge_regression_page},
        {"title": "Lasso Regression", "function": lasso_regression_page},
        {"title": "ElasticNet Regression", "function": elasticnet_regression_page},
        {"title": "Decision Tree Regressor", "function": decision_tree_regression_page},
        {"title": "Random Forest Regressor", "function": random_forest_regression_page},
        {"title": "Gradient Boosting Regressor", "function": gradient_boosting_regression_page},
        {"title": "XGBoost Regressor", "function": xgboost_regression_page},
        {"title": "Support Vector Regression (SVR)", "function": svr_regression_page},
        {"title": "Neural Net Regressor (MLP)", "function": mlp_regressor_page}
    ],
    "Unsupervised Learning": [
        {"title": "Unsupervised Clustering", "function": clustering_comparison_page},
        {"title": "Dimensionality Reduction", "function": dimensionality_reduction_explorer},
        {"title": "Self-Organizing Map Clustering", "function": som_clustering_page}
    ],
    "Reinforcement Learning": [
        {"title": "Basic Q Learning", "function": q_learning_frozenlake_page},
        {"title": "Multi Armed Bandit", "function": multi_armed_bandit_page},
        {"title": "Simple Policy Gradient", "function": policy_gradient_cartpole_page},
        {"title": "Swarming Behaviour", "function": swarming_behavior_page}
    ],
    "Anomaly Detection": [
        {"title": "Anomaly Detection with Isolation Forest", "function": isolation_forest_page}
    ],
    "Sensor Fusion": [
        {"title": "Complementary Filter Sensor Fusion", "function": complementary_filter_sensor_fusion_page},
        {"title": "Kalman Filter Sensor Fusion", "function": kalman_filter_sensor_fusion_page},
        {"title": "Particle Filter Sensor Fusion", "function": particle_filter_sensor_fusion_page}
    ],
    "Emergent Behaviour": [
        {"title": "Boids Flocking Simulation", "function": boids_flocking_page},
        {"title": "Game of Life", "function": game_of_life_page},
        {"title": "Swarm Consensus", "function": particle_swarming_page},
        {"title": "Leader-Follower Control", "function": leader_follower_page},
        {"title": "Goal-Based Attractor", "function": attractor_control_page},
        {"title": "Parameter Modulation", "function": parameter_modulation_page}
    ],
    "AI Certification (DO-178C)": [
        {"title": "DO-178C AI Certification Suite", "function": do178c_ai_certification_page}
    ]
}

# ---- Sidebar Navigation ----
st.sidebar.title("Navigation")
category = st.sidebar.selectbox("Choose a Category", list(categories.keys()))
page_title = st.sidebar.selectbox(
    "Select an Example",
    [page['title'] for page in categories[category]]
)

# Display the selected page
for page in categories[category]:
    if page['title'] == page_title:
        page['function']()
