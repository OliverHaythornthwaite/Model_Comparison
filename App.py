import streamlit as st

# Import your page functions (assuming you have these as functions in separate files or defined below)
from Classifier_Comparison_App import classification_comparison_page
from Regression_Comparison_App import regression_comparison_page
from Clustering_Comparison_App import clustering_comparison_page
from Dimensionality_Reduction_Comparison_App import dimensionality_reduction_explorer
from Self_Organising_Maps_Comparison_App import som_clustering_page
from Anomoly_Detection_Comparison_App import isolation_forest_page
from Basic_Q_Learning_Comparison_App import q_learning_frozenlake_page
from Multi_Armed_Bandit_Comparison_App import multi_armed_bandit_page
from Simple_Policy_Gradient_App import policy_gradient_cartpole_page
from Swarming_Behaviour_App import swarming_behavior_page

st.set_page_config(page_title="ML Model Comparison Suite", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Classification Model Comparison",
    "Regression Model Comparison",
    "Unsupervised Clustering",
    "Dimensionality Reduction",
    "Self-Organizing Map Clustering",
    "Anomoly Detection",
    "Basic Q Learning",
    "Multi Armed Bandit",
    "Simple Policy Gradient",
    "Swarming Behaviour"
])

if page == "Classification Model Comparison":
    classification_comparison_page()
elif page == "Regression Model Comparison":
    regression_comparison_page()
elif page == "Unsupervised Clustering":
    clustering_comparison_page()
elif page == "Dimensionality Reduction":
    dimensionality_reduction_explorer()
elif page == "Self-Organizing Map Clustering":
    som_clustering_page()
elif page == "Anomoly Detection":
    isolation_forest_page()
elif page == "Basic Q Learning":
    q_learning_frozenlake_page()
elif page == "Multi Armed Bandit":
    multi_armed_bandit_page()
elif page == "Simple Policy Gradient":
    policy_gradient_cartpole_page()
elif page == "Swarming Behaviour":
    swarming_behavior_page()