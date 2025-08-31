# CONWAY'S GAME OF LIFE
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def game_of_life_page():
    st.title("🟩 Conway's Game of Life")

    st.markdown("""
    **Game of Life** is a cellular automaton where emergent patterns appear from simple rules:
    - Any live cell with 2 or 3 neighbors survives
    - Any dead cell with 3 neighbors becomes alive
    - All other cells die or remain dead
    """)

    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    grid_size = st.sidebar.slider("Grid Size", 10, 100, 50)
    n_steps = st.sidebar.slider("Number of Steps", 10, 200, 50)
    density = st.sidebar.slider("Initial Live Cell Density", 0.1, 0.9, 0.3)

    # Initialize grid
    np.random.seed(42)
    grid = np.random.rand(grid_size, grid_size) < density

    fig, ax = plt.subplots()
    for _ in range(n_steps):
        ax.clear()
        ax.imshow(grid, cmap='Greens')
        ax.set_title("Conway's Game of Life")
        # Compute neighbors
        neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                        for i in (-1,0,1) for j in (-1,0,1) if (i,j) != (0,0))
        # Apply rules
        grid = (neighbors == 3) | (grid & (neighbors == 2))
    st.pyplot(fig)
