import streamlit as st
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def policy_gradient_cartpole_page():
    st.title("Reinforcement Learning: Simple Policy Gradient")

    with st.expander("❓ What is the Simple Policy Gradient Method?"):
        st.markdown("""
        ### Explanation of the Simple Policy Gradient Method

        The **Policy Gradient** approach is a fundamental method in reinforcement learning where the agent learns a **parameterized policy** directly, rather than estimating value functions.

        #### Core Idea

        Instead of learning a value function to indirectly derive a policy, policy gradient methods optimize the policy parameters by **maximizing expected cumulative reward** through gradient ascent.

        #### How It Works

        - The policy \(\pi_\theta(a|s)\) defines the probability of taking action \(a\) in state \(s\), parameterized by \(\theta\).
        - The goal is to find parameters \(\theta\) that maximize the expected reward.
        - The agent collects trajectories by interacting with the environment using the current policy.
        - Using these trajectories, the policy parameters are updated in the direction that increases the likelihood of actions leading to higher rewards.

        #### Advantages

        - Can learn stochastic policies naturally.
        - Works well in continuous action spaces.
        - Does not require a value function (though can be combined with one).
        - Suitable for high-dimensional or complex action spaces.

        #### Challenges

        - High variance in gradient estimates can lead to unstable training.
        - Requires careful tuning of learning rates and exploration.
        - Sample inefficient compared to some value-based methods.

        #### Why Policy Gradients Matter

        Policy gradients provide a direct and flexible way to optimize policies and form the basis for advanced algorithms like **REINFORCE**, **Actor-Critic methods**, and **Proximal Policy Optimization (PPO)**.

        #### Summary

        In simple terms, the policy gradient method helps an agent learn how to **choose actions directly** by tweaking its policy parameters to get better rewards over time, using feedback from its own experience.

        ---

        This approach is especially useful when the action space is large or continuous and when we want the agent to learn stochastic policies that can better handle uncertainty and variability in environments.
        """)

    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    st.write("This demo runs a very simple policy gradient with a linear policy.")

    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
    episodes = st.sidebar.slider("Episodes", 100, 2000, 500)

    # Linear policy parameters
    theta = np.random.rand(n_states, n_actions) * 0.01

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    rewards_history = []
    avg_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        states, actions, rewards = [], [], []

        while not done:
            logits = np.dot(state, theta)
            probs = softmax(logits)
            action = np.random.choice(n_actions, p=probs)

            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

        rewards_history.append(episode_reward)

        # Compute discounted rewards
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0
        gamma = 0.99
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + gamma * cumulative
            discounted_rewards[t] = cumulative

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)

        # Policy gradient update
        for i in range(len(states)):
            s = states[i]
            a = actions[i]
            d_r = discounted_rewards[i]
            probs = softmax(np.dot(s, theta))
            grad = -probs
            grad[a] += 1
            theta += learning_rate * d_r * np.outer(s, grad)

        avg_rewards.append(np.mean(rewards_history[-50:]))

        if episode % 50 == 0:
            st.write(f"Episode {episode}, Avg Reward (last 50): {avg_rewards[-1]:.2f}")

    # Plot rewards
    fig, ax = plt.subplots()
    ax.plot(rewards_history, label="Episode Reward")
    ax.plot(avg_rewards, label="Average Reward (50 episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    st.pyplot(fig)