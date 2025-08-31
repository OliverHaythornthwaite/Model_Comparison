# COMPLEMENTARY FILTER SENSOR FUSION DEMO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def complementary_filter_sensor_fusion_page():
    st.title("🔗 Complementary Filter Sensor Fusion Demo")

    st.markdown("""
    **Complementary Filter** combines sensors with different characteristics:
    - Example: accelerometer (low-frequency) + gyroscope (high-frequency)
    - Uses a weighted combination to reduce noise
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    alpha = st.sidebar.slider("Alpha (weight for gyro)", 0.0, 1.0, 0.98)
    n_steps = st.sidebar.slider("Number of Time Steps", 10, 200, 50)
    np.random.seed(42)

    # Simulate signals
    true_angle = np.cumsum(np.random.randn(n_steps) * 0.1)
    gyro_signal = true_angle + np.random.normal(0, 0.05, n_steps)
    accel_signal = true_angle + np.random.normal(0, 0.2, n_steps)

    # Complementary filter
    estimates = [true_angle[0]]
    for i in range(1, n_steps):
        estimate = alpha * (estimates[-1] + gyro_signal[i] - gyro_signal[i-1]) + (1-alpha) * accel_signal[i]
        estimates.append(estimate)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(true_angle, label="True Angle", linewidth=2)
    ax.plot(accel_signal, label="Accelerometer", alpha=0.5)
    ax.plot(gyro_signal, label="Gyroscope", alpha=0.5)
    ax.plot(estimates, label="Complementary Filter Estimate", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Angle")
    ax.legend()
    st.pyplot(fig)
