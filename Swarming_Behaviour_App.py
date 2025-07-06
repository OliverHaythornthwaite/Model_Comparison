import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

def swarming_behavior_page():
    st.title("Swarming Behavior: Boids Model with Collision Highlight")

    with st.expander("ℹ️ About Swarming Behavior (Boids Model)"):
        st.markdown("""
        The **Boids model** simulates swarming behavior using simple local rules:
        - **Alignment**: Match the direction of nearby boids.
        - **Cohesion**: Move toward the average position of neighbors.
        - **Separation**: Steer away from nearby boids to avoid crowding.

        In this version, if two boids come **too close (within 2 units)**, they are highlighted in **red** to indicate a potential collision.
        """)

    # Sidebar
    n_boids = st.sidebar.slider("Number of Boids", 10, 200, 50)
    perception = st.sidebar.slider("Perception Radius", 5, 100, 30)
    max_speed = st.sidebar.slider("Max Speed", 1, 10, 4)
    max_force = st.sidebar.slider("Max Steering Force", 0.01, 1.0, 0.05)
    n_frames = st.sidebar.slider("Animation Frames", 10, 300, 100)
    collision_distance = 2.0  # distance threshold for "touching"

    # Initialize
    positions = np.random.rand(n_boids, 2) * 100
    velocities = (np.random.rand(n_boids, 2) - 0.5) * 10

    def limit_vector(vec, max_val):
        norm = np.linalg.norm(vec)
        return vec / norm * max_val if norm > max_val else vec

    def update_boids(positions, velocities):
        new_velocities = np.copy(velocities)
        for i in range(n_boids):
            pos = positions[i]
            neighbors = []
            for j in range(n_boids):
                if i != j and np.linalg.norm(positions[j] - pos) < perception:
                    neighbors.append(j)

            if neighbors:
                avg_vel = np.mean(velocities[neighbors], axis=0)
                steer_align = limit_vector(avg_vel - velocities[i], max_force)

                avg_pos = np.mean(positions[neighbors], axis=0)
                steer_cohesion = limit_vector(avg_pos - pos, max_force)

                steer_separation = np.zeros(2)
                for j in neighbors:
                    diff = pos - positions[j]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        steer_separation += diff / (dist**2)
                steer_separation = limit_vector(steer_separation, max_force)

                steering = steer_align + steer_cohesion + steer_separation
                new_velocities[i] += steering
                new_velocities[i] = limit_vector(new_velocities[i], max_speed)

        new_positions = (positions + new_velocities) % 100
        return new_positions, new_velocities

    # Animation loop
    plot_area = st.empty()
    for _ in range(n_frames):
        positions, velocities = update_boids(positions, velocities)

        # Collision detection
        colors = ['blue'] * n_boids
        for i in range(n_boids):
            for j in range(i + 1, n_boids):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < collision_distance:
                    colors[i] = 'red'
                    colors[j] = 'red'

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, edgecolors='k')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Boids Swarming (Red = Too Close)")
        plot_area.pyplot(fig)

        time.sleep(0.05)