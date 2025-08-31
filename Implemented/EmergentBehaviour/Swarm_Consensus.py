# PARTICLE SWARMING / CONSENSUS
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def particle_swarming_page():
    st.title("🌀 Particle Swarming / Consensus Simulation")

    st.markdown("""
    **Swarming / Consensus** behavior emerges from local interactions:
    - Each agent adjusts its position based on neighbors
    - Leads to collective motion or agreement without central control
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_particles = st.sidebar.slider("Number of Particles", 10, 200, 50)
    n_steps = st.sidebar.slider("Number of Steps", 10, 100, 30)
    influence_radius = st.sidebar.slider("Influence Radius", 1, 20, 5)

    # Initialize positions
    np.random.seed(42)
    positions = np.random.rand(n_particles, 2) * 100

    # Swarm update function
    def update(positions):
        new_positions = positions.copy()
        for i in range(n_particles):
            neighbors = np.linalg.norm(positions - positions[i], axis=1) < influence_radius
            if np.sum(neighbors) > 1:
                center = np.mean(positions[neighbors], axis=0)
                new_positions[i] += (center - positions[i]) * 0.1
        return new_positions

    # Simulate
    fig, ax = plt.subplots()
    for _ in range(n_steps):
        positions = update(positions)
        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], c='orange')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Particle Swarming / Consensus")
    st.pyplot(fig)
