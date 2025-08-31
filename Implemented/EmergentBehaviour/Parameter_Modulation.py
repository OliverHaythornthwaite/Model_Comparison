# PARAMETER MODULATION / RULE TUNING DEMO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def parameter_modulation_page():
    st.title("⚙️ Parameter Modulation to Control Emergent Behavior")

    st.markdown("""
    **Parameter Modulation** adjusts local agent rules to influence global patterns:
    - Changing cohesion, separation, or alignment weights
    - Emergent behavior can become more clustered, dispersed, or aligned
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_agents = st.sidebar.slider("Number of Agents", 10, 200, 50)
    n_steps = st.sidebar.slider("Number of Steps", 10, 100, 30)
    w_cohesion = st.sidebar.slider("Cohesion Weight", 0.0, 1.0, 0.05)
    w_alignment = st.sidebar.slider("Alignment Weight", 0.0, 1.0, 0.05)
    w_separation = st.sidebar.slider("Separation Weight", 0.0, 1.0, 0.05)
    speed_limit = st.sidebar.slider("Max Speed", 0.1, 5.0, 1.0)

    # Initialize positions and velocities
    np.random.seed(42)
    positions = np.random.rand(n_agents, 2) * 100
    velocities = (np.random.rand(n_agents, 2) - 0.5) * speed_limit

    # Update function
    def update(positions, velocities):
        for i in range(n_agents):
            # Cohesion
            center = np.mean(positions, axis=0)
            cohesion = (center - positions[i]) * w_cohesion
            # Alignment
            alignment = (np.mean(velocities, axis=0) - velocities[i]) * w_alignment
            # Separation
            diff = positions[i] - positions
            dist = np.linalg.norm(diff, axis=1)
            separation = np.sum(diff[dist < 5], axis=0) * w_separation if np.any(dist < 5) else 0
            # Update velocity
            velocities[i] += cohesion + alignment + separation
            speed = np.linalg.norm(velocities[i])
            if speed > speed_limit:
                velocities[i] = velocities[i] / speed * speed_limit
        positions += velocities
        return positions, velocities

    # Simulate
    fig, ax = plt.subplots()
    for _ in range(n_steps):
        positions, velocities = update(positions, velocities)
        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], c='purple')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Parameter Modulation Control")
    st.pyplot(fig)
