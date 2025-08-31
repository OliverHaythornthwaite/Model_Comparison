# LEADER-FOLLOWER CONTROL DEMO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def leader_follower_page():
    st.title("🧭 Leader-Follower Control of Emergent Behavior")

    st.markdown("""
    **Leader-Follower Control** guides a swarm by designating leader agents:
    - Leaders follow a target trajectory
    - Followers adjust positions relative to leaders
    - Emergent behavior becomes steerable
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_agents = st.sidebar.slider("Number of Agents", 10, 200, 50)
    n_leaders = st.sidebar.slider("Number of Leaders", 1, 5, 1)
    n_steps = st.sidebar.slider("Number of Steps", 10, 100, 30)
    influence_radius = st.sidebar.slider("Influence Radius", 1, 20, 5)

    # Initialize positions
    np.random.seed(42)
    positions = np.random.rand(n_agents, 2) * 100
    leaders = np.random.choice(range(n_agents), n_leaders, replace=False)

    # Leader trajectory
    target = np.array([80, 80])

    def update(positions):
        new_positions = positions.copy()
        for i in range(n_agents):
            if i in leaders:
                # Leaders move toward target
                new_positions[i] += (target - positions[i]) * 0.1
            else:
                # Followers move toward neighbors
                neighbors = np.linalg.norm(positions - positions[i], axis=1) < influence_radius
                if np.sum(neighbors) > 1:
                    center = np.mean(positions[neighbors], axis=0)
                    new_positions[i] += (center - positions[i]) * 0.05
        return new_positions

    # Simulate
    fig, ax = plt.subplots()
    for _ in range(n_steps):
        positions = update(positions)
        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Agents')
        ax.scatter(positions[leaders, 0], positions[leaders, 1], c='red', label='Leaders')
        ax.scatter(target[0], target[1], c='green', marker='*', s=200, label='Target')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Leader-Follower Control")
        ax.legend()
    st.pyplot(fig)
