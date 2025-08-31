# KALMAN FILTER SENSOR FUSION DEMO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def kalman_filter_sensor_fusion_page():
    st.title("🛰️ Kalman Filter Sensor Fusion Demo")

    st.markdown("""
    **Kalman Filter** is an optimal recursive estimator:
    - Combines multiple noisy measurements
    - Predicts state over time
    - Widely used in robotics, navigation, and sensor fusion
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_steps = st.sidebar.slider("Number of Time Steps", 10, 200, 50)
    process_var = st.sidebar.slider("Process Variance", 0.001, 1.0, 0.1)
    measurement_var = st.sidebar.slider("Measurement Variance", 0.01, 5.0, 1.0)

    # Simulate true signal and noisy measurements
    np.random.seed(42)
    true_signal = np.cumsum(np.random.randn(n_steps))  # Random walk
    measurements = true_signal + np.random.normal(0, np.sqrt(measurement_var), n_steps)

    # Kalman filter
    x_est = 0.0
    P = 1.0
    Q = process_var
    R = measurement_var
    estimates = []

    for z in measurements:
        # Prediction step
        P = P + Q
        # Update step
        K = P / (P + R)
        x_est = x_est + K * (z - x_est)
        P = (1 - K) * P
        estimates.append(x_est)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(true_signal, label="True Signal", linewidth=2)
    ax.plot(measurements, 'o', label="Measurements", alpha=0.5)
    ax.plot(estimates, label="Kalman Estimate", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)
