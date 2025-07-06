import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def swarming_behavior_page():
    st.title("Swarming Behavior: Boids Model Simulation")

    # Description in a hideable expander
    with st.expander("ℹ️ About Swarming Behavior (Boids Model)"):
        st.markdown("""
        **Swarming behavior** is a fascinating natural phenomenon where individual agents, such as birds, fish, or insects, collectively move in coordinated patterns without centralized control. The Boids model, developed by Craig Reynolds in 1986, simulates this by programming each agent (called a "boid") to follow three simple local rules:

        1. **Alignment** — steer towards the average heading (direction) of nearby flockmates.  
        2. **Cohesion** — steer to move closer to the average position of neighbors, promoting group unity.  
        3. **Separation** — steer away to avoid crowding and collisions with nearby boids.

        Despite the simplicity of these rules, the interactions produce complex, lifelike flocking or swarming behavior that resembles natural phenomena seen in bird flocks or fish schools. This emergent behavior results from local agent interactions, making it a powerful example of decentralized collective intelligence.

        The simulation visualizes these principles by showing how a group of boids move dynamically across a space, adjusting their velocities based on their neighbors’ positions and velocities. By tuning parameters like perception radius and speed, you can explore how different conditions affect swarm cohesion and movement patterns.
        """)

    # Parameters
    n_boids = st.sidebar.slider("Number of Boids", 20, 200, 100)
    perception = st.sidebar.slider("Perception Radius", 10, 100, 50)
    max_speed = st.sidebar.slider("Max Speed", 1, 10, 4)
    max_force = st.sidebar.slider("Max Steering Force", 0.01, 0.5, 0.1)

    # Initialize boids with random positions and velocities
    positions = np.random.rand(n_boids, 2) * 100
    velocities = (np.random.rand(n_boids, 2) - 0.5) * 2

    def limit_vector(vec, max_val):
        norm = np.linalg.norm(vec)
        if norm > max_val:
            return vec / norm * max_val
        return vec

    def step(positions, velocities):
        new_velocities = np.zeros_like(velocities)
        for i, pos in enumerate(positions):
            neighbors = []
            for j, other_pos in enumerate(positions):
                if i != j and np.linalg.norm(pos - other_pos) < perception:
                    neighbors.append(j)
            if neighbors:
                # Alignment
                avg_vel = np.mean(velocities[neighbors], axis=0)
                steer_align = avg_vel - velocities[i]
                steer_align = limit_vector(steer_align, max_force)

                # Cohesion
                avg_pos = np.mean(positions[neighbors], axis=0)
                direction_to_center = avg_pos - pos
                steer_cohesion = limit_vector(direction_to_center, max_force)

                # Separation
                separation_vec = np.zeros(2)
                for j in neighbors:
                    diff = pos - positions[j]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        separation_vec += diff / dist**2
                steer_separation = limit_vector(separation_vec, max_force)

                # Combine forces
                steering = steer_align + steer_cohesion + steer_separation
            else:
                steering = np.zeros(2)

            new_velocity = velocities[i] + steering
            new_velocity = limit_vector(new_velocity, max_speed)
            new_velocities[i] = new_velocity

        new_positions = positions + new_velocities

        # Wrap around edges
        new_positions = new_positions % 100

        return new_positions, new_velocities

    # Animate the boids for a fixed number of frames
    frames = 100

    fig, ax = plt.subplots(figsize=(8, 8))
    scat = ax.scatter(positions[:, 0], positions[:, 1], color='blue')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Boids Swarming Simulation")

    for _ in range(frames):
        positions, velocities = step(positions, velocities)

    scat.set_offsets(positions)
    st.pyplot(fig)