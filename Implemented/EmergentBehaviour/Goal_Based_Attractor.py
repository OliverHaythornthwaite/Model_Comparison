# ATTRACTOR / GOAL-BASED CONTROL DEMO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def attractor_control_page():
    st.title("🎯 Attractor / Goal-Based Control of Emergent Behavior")

    st.markdown("""
    **Attractor-Based Control** guides a swarm or agents using goal points:
    - Agents move toward attractors
    - Emergent patterns are shaped by attractor positions
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_agents = st.sidebar.slider("Number of Agents", 10, 200, 50)
    n_steps = st.sidebar.slider("Number of Steps", 10, 100, 30)
    n_attractors = st.sidebar.slider("Number of Attractors", 1, 5, 2)
    influence_radius = st.sidebar.slider("Influence Radius", 1, 20, 5)

    # Initialize positions
    np.random.seed(42)
    positions = np.random.rand(n_agents, 2) * 100
    attractors = np.random.rand(n_attractors, 2) * 100

    def update(positions):
        new_positions = positions.copy()
        for i in range(n_agents):
            # Move toward nearest attractor
            distances = np.linalg.norm(attractors - positions[i], axis=1)
            closest = attractors[np.argmin(distances)]
            new_positions[i] += (closest - positions[i]) * 0.05
        return new_positions

    # Simulate
    fig, ax = plt.subplots()
    for _ in range(n_steps):
        positions = update(positions)
        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], c='orange', label='Agents')
        ax.scatter(attractors[:, 0], attractors[:, 1], c='green', marker='*', s=200, label='Attractors')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Attractor-Based Control")
        ax.legend()
    st.pyplot(fig)
