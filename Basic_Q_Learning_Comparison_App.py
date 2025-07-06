import streamlit as st
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import gym
import matplotlib.pyplot as plt
import seaborn as sns

def q_learning_frozenlake_page():
    st.title("Reinforcement Learning: Q-Learning on FrozenLake")

    with st.expander("❓ What is the FrozenLake Problem?"):
        st.markdown("""
        ### Explanation of the FrozenLake Problem

        The **FrozenLake** environment is a classic reinforcement learning benchmark provided by OpenAI Gym. It models a grid world where an agent must navigate from a starting point to a goal, avoiding holes in the ice. The environment is typically represented as a 4x4 grid:

        - **Start (S)**: The starting cell where the agent begins.
        - **Frozen (F)**: Safe ice cells where the agent can walk.
        - **Hole (H)**: Dangerous cells that cause the agent to fall through and lose the episode.
        - **Goal (G)**: The target cell that the agent tries to reach.

        #### Objective

        The agent's goal is to find a path from the **Start (S)** to the **Goal (G)** by moving one step at a time in any of four directions: **Left, Down, Right, Up**. The environment provides a reward of **+1** if the agent reaches the goal and **0** otherwise. The episode ends either when the agent reaches the goal or falls into a hole.

        #### Environment Dynamics

        - The environment can be **deterministic** (`is_slippery=False`) or **stochastic/slippery** (`is_slippery=True`).  
        - In the slippery version, the agent may slide in unintended directions, making the problem more challenging.  
        - In the deterministic version, the agent moves exactly as intended.

        #### Why is FrozenLake Important?

        FrozenLake is a simple yet powerful environment for testing reinforcement learning algorithms because:

        - It involves **sequential decision-making** under uncertainty (especially in the slippery version).
        - It requires the agent to learn a **policy** mapping states to optimal actions.
        - It highlights the challenges of balancing **exploration and exploitation**.
        - Despite its simplicity, it illustrates key concepts like **value functions**, **Q-learning**, and **policy learning**.

        #### State and Action Space

        - The state space consists of the 16 discrete positions on the grid (0 to 15).  
        - The action space consists of 4 discrete actions:  
          - 0 = Left  
          - 1 = Down  
          - 2 = Right  
          - 3 = Up

        #### Challenges for the Agent

        - Avoiding holes that terminate the episode with no reward.
        - Learning the shortest or safest path to the goal.
        - Managing exploration to find the best route.
        - Dealing with stochastic transitions (if slippery).

        ---

        By training a Q-learning agent on FrozenLake, we aim to learn an optimal policy that guides the agent safely to the goal while maximizing cumulative reward. The agent updates its knowledge (Q-values) about state-action pairs over many episodes, balancing immediate and future rewards.
        """)

    env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.8)
    gamma = st.sidebar.slider("Discount Factor (gamma)", 0.0, 1.0, 0.95)
    epsilon = st.sidebar.slider("Exploration Rate (epsilon)", 0.0, 1.0, 0.1)
    episodes = st.sidebar.slider("Training Episodes", 100, 10000, 2000)

    def choose_action(state):
        if np.random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()  # unpack tuple from reset
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)  # unpack 5 values
            done = done or truncated

            old_value = Q[state, action]
            next_max = np.max(Q[next_state])

            Q[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    st.write(f"Training completed over {episodes} episodes")

    # Show reward trend
    st.subheader("Episode Rewards Over Time")
    st.line_chart(rewards)

    # Show reward statistics
    mean_reward = np.mean(rewards)
    st.write(f"Average Reward: {mean_reward:.3f}")

    # Plot heatmaps of Q-values
    st.subheader("Q-values Heatmap per Action")
    actions = ["Left ←", "Down ↓", "Right →", "Up ↑"]

    fig, axes = plt.subplots(1, n_actions, figsize=(20, 4))
    for i in range(n_actions):
        sns.heatmap(Q[:, i].reshape((4, 4)), annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i], cbar=i == n_actions-1)
        axes[i].set_title(f"Action: {actions[i]}")
        axes[i].invert_yaxis()
        axes[i].set_xlabel("Column")
        axes[i].set_ylabel("Row")
    st.pyplot(fig)

    # State-value heatmap (max Q-value per state)
    st.subheader("State-Value Function Heatmap (Max Q-value per State)")
    V = np.max(Q, axis=1)
    fig2, ax2 = plt.subplots()
    sns.heatmap(V.reshape((4,4)), annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    st.pyplot(fig2)

    # Display policy with arrows in a colored grid
    st.subheader("Learned Policy Grid")

    arrows = ["←", "↓", "→", "↑"]
    policy_grid = [arrows[np.argmax(Q[i])] for i in range(n_states)]
    grid_size = int(np.sqrt(n_states))

    # Color map for actions
    action_colors = {
        "←": "#1f77b4",  # blue
        "↓": "#ff7f0e",  # orange
        "→": "#2ca02c",  # green
        "↑": "#d62728"   # red
    }

    policy_display = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            action = policy_grid[i * grid_size + j]
            color = action_colors[action]
            # Use st.markdown with HTML for colored arrows
            row.append(f"<span style='color:{color}; font-size: 28px;'>{action}</span>")
        policy_display.append(" ".join(row))

    for row_html in policy_display:
        st.markdown(row_html, unsafe_allow_html=True)