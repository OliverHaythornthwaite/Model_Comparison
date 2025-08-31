# BOIDS / FLOCKING SIMULATION
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def boids_flocking_page():
    st.title("🦅 Boids / Flocking Simulation")

    st.markdown("""
    **Flocking behavior** emerges from three simple rules for each agent (boid):
    1. **Separation** – avoid crowding neighbors  
    2. **Alignment** – match velocity with nearby boids  
    3. **Cohesion** – move towards the average position of neighbors  
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_boids = st.sidebar.slider("Number of Boids", 10, 200, 50)
    n_steps = st.sidebar.slider("Number of Time Steps", 10, 100, 30)
    speed_limit = st.sidebar.slider("Max Speed", 0.5, 5.0, 2.0)

    # Initialize positions and velocities
    np.random.seed(42)
    positions = np.random.rand(n_boids, 2) * 100
    velocities = (np.random.rand(n_boids, 2) - 0.5) * speed_limit

    # Simple flocking update function
    def update(positions, velocities):
        for i in range(n_boids):
            # Compute vector to center of mass
            center = np.mean(positions, axis=0)
            cohesion = (center - positions[i]) * 0.01
            # Compute alignment
            alignment = (np.mean(velocities, axis=0) - velocities[i]) * 0.05
            # Compute separation
            diff = positions[i] - positions
            dist = np.linalg.norm(diff, axis=1)
            separation = np.sum(diff[dist < 5], axis=0) * 0.05 if np.any(dist < 5) else 0
            # Update velocity and clip
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
        ax.scatter(positions[:, 0], positions[:, 1], c='blue')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Flocking Simulation")
    st.pyplot(fig)
