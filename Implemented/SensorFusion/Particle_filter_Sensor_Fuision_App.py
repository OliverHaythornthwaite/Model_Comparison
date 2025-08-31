# PARTICLE FILTER SENSOR FUSION DEMO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def particle_filter_sensor_fusion_page():
    st.title("🔮 Particle Filter Sensor Fusion Demo")

    st.markdown("""
    **Particle Filter** is a probabilistic sensor fusion method:
    - Uses a set of particles to represent the probability distribution of the state
    - Each particle is weighted by likelihood based on sensor measurements
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_particles = st.sidebar.slider("Number of Particles", 50, 500, 100)
    n_steps = st.sidebar.slider("Number of Time Steps", 10, 100, 30)
    measurement_var = st.sidebar.slider("Measurement Variance", 0.01, 2.0, 0.2)

    # True state
    np.random.seed(42)
    true_state = np.cumsum(np.random.randn(n_steps))

    # Particle filter initialization
    particles = np.random.randn(n_particles)
    estimates = []

    for t in range(n_steps):
        # Predict (add small process noise)
        particles += np.random.normal(0, 0.1, n_particles)
        # Measurement update
        weights = np.exp(-0.5 * ((particles - true_state[t])**2) / measurement_var)
        weights /= np.sum(weights)
        # Resample
        indices = np.random.choice(range(n_particles), size=n_particles, p=weights)
        particles = particles[indices]
        # Estimate
        estimates.append(np.mean(particles))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(true_state, label="True State", linewidth=2)
    ax.plot(estimates, label="Particle Filter Estimate", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)
