import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def multi_armed_bandit_page():
    st.title("Reinforcement Learning: Multi-Armed Bandit Problem")

    with st.expander("❓ What is the Multi-Armed Bandit Problem?"):
        st.markdown("""
        ### Explanation of the Multi-Armed Bandit Problem

        The **Multi-Armed Bandit (MAB)** problem is a foundational problem in reinforcement learning and decision theory. Imagine you're in a casino facing multiple slot machines (or "one-armed bandits"). Each slot machine has an unknown probability distribution of payouts (rewards).

        #### Objective

        Your goal is to maximize the total reward over a sequence of lever pulls by deciding which machine to play at each step.

        #### Key Challenges

        - **Exploration vs. Exploitation**:  
          You need to balance **exploration** (trying different slot machines to learn their payout rates) and **exploitation** (choosing the machine you currently believe to be the best to maximize reward).

        - **Uncertainty and Learning**:  
          You start without knowing the reward probabilities for any machine. Your strategy should learn from past outcomes to improve future decisions.

        #### Formal Setup

        - You have **k** slot machines (arms), each with an unknown reward distribution.
        - At each time step, you select one arm to pull and observe the reward.
        - The goal is to maximize the cumulative reward over time.

        #### Applications

        - Online advertising (choosing which ad to display).
        - Clinical trials (selecting the best treatment).
        - Recommendation systems (suggesting products or content).
        - Any scenario involving sequential decision-making with uncertainty.

        #### Common Algorithms

        - **ε-greedy**: Mostly choose the best-known arm but occasionally explore.
        - **Upper Confidence Bound (UCB)**: Select arms based on optimism in the face of uncertainty.
        - **Thompson Sampling**: Bayesian approach sampling from posterior distributions.

        #### Why is MAB Important?

        Despite its simplicity, the Multi-Armed Bandit problem captures the essence of the exploration-exploitation trade-off, a core challenge in reinforcement learning and adaptive systems.

        ---

        By solving the Multi-Armed Bandit problem, we learn strategies that can be generalized to more complex reinforcement learning tasks where agents must make decisions under uncertainty to maximize rewards over time.
        """)

    n_arms = st.sidebar.slider("Number of Arms", 2, 10, 5)
    epsilon = st.sidebar.slider("Epsilon (exploration rate)", 0.0, 1.0, 0.1)
    episodes = st.sidebar.slider("Episodes", 100, 5000, 1000)

    # True reward probabilities (unknown to agent)
    true_rewards = np.random.rand(n_arms)

    # Initialize estimates and counts
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)

    rewards = []

    for ep in range(episodes):
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(Q)

        reward = np.random.binomial(1, true_rewards[action])
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)

    st.write(f"True probabilities of rewards per arm: {np.round(true_rewards, 2)}")
    st.write(f"Estimated values: {np.round(Q, 2)}")

    fig, ax = plt.subplots()
    ax.plot(np.cumsum(rewards), label="Cumulative Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    st.pyplot(fig)