from env import GridWorld
from agent import QAgent
from rule_agent import RuleBasedAgent
from train import train

from dqn_agent import DQNAgent
from train_dqn import train_dqn

from md_dqn_agent import MDDQNAgent
from train_md_dqn import train_md_dqn

import matplotlib.pyplot as plt
import numpy as np


def clone_env(base_env, base_grid):
    env = GridWorld(
        size=base_env.size,
        num_obstacles=base_env.num_obstacles,
        num_resources=base_env.num_resources,
        num_chargers=base_env.num_chargers
    )

    env.grid = base_grid.copy()
    env.original_grid = base_grid.copy()

    env.agent_pos = (0, 0)
    env.energy = 100
    env.resources_collected = 0
    env.steps = 0
    env.collisions = 0
    env.total_reward = 0.0

    return env


def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')


if __name__ == "__main__":

    print("Creating base map...")
    base_env = GridWorld()
    base_grid = base_env.original_grid.copy()

    # create envs
    rule_env = clone_env(base_env, base_grid)
    rl_env = clone_env(base_env, base_grid)
    dqn_env = clone_env(base_env, base_grid)
    md_dqn_env = clone_env(base_env, base_grid)

    # -----------------------------
    # Rule-Based Agent
    # -----------------------------
    print("\nRunning Rule-Based Agent...")
    rule_agent = RuleBasedAgent()

    while True:
        action = rule_agent.select_action(rule_env)
        _, _, done = rule_env.step(action)

        if done or rule_env.steps >= 200:
            break

    print(f"Rule-Based | Resources: {rule_env.resources_collected} | "
          f"Collisions: {rule_env.collisions} | Steps: {rule_env.steps}")

    # -----------------------------
    # Q-Learning
    # -----------------------------
    print("\nTraining Q-Learning Agent...")
    rl_agent = QAgent()
    rl_env, rl_rewards = train(rl_agent, episodes=300, env=rl_env)

    print("\nRunning Q-Learning Agent final episode...")
    rl_env.soft_reset()
    state = rl_env.get_state()

    while True:
        action = rl_agent.choose_action(state)
        state, _, done = rl_env.step(action)
        if done or rl_env.steps >= 200:
            break

    # -----------------------------
    # DQN
    # -----------------------------
    print("\nTraining DQN Agent...")
    dqn_agent = DQNAgent()
    dqn_env, dqn_rewards = train_dqn(dqn_agent, episodes=300, env=dqn_env)

    print("\nRunning DQN Agent final episode...")
    dqn_env.soft_reset()
    state = dqn_agent.featurize_state(dqn_env)

    while True:
        action = dqn_agent.choose_action(state, greedy=True)
        _, _, done = dqn_env.step(action)
        state = dqn_agent.featurize_state(dqn_env)
        if done or dqn_env.steps >= 200:
            break

    dqn_agent.save_weights("source_dqn.pth")

    # -----------------------------
    # MD-DQN
    # -----------------------------
    print("\nTraining MD-DQN Agent...")
    md_dqn_agent = MDDQNAgent()
    md_dqn_agent.load_source_weights("source_dqn.pth")
    md_dqn_env, md_dqn_rewards = train_md_dqn(md_dqn_agent, episodes=300, env=md_dqn_env)

    print("\nRunning MD-DQN Agent final episode...")
    md_dqn_env.soft_reset()
    state = md_dqn_agent.featurize_state(md_dqn_env)

    while True:
        action = md_dqn_agent.choose_action(state, greedy=True)
        _, _, done = md_dqn_env.step(action)
        state = md_dqn_agent.featurize_state(md_dqn_env)
        if done or md_dqn_env.steps >= 200:
            break

    # -----------------------------
    # FINAL COMPARISON
    # -----------------------------
    print("\n-*-*- FINAL COMPARISON -*-*-")

    agents = ["Rule", "Q-Learning", "DQN", "MD-DQN"]

    resources = [
        rule_env.resources_collected,
        rl_env.resources_collected,
        dqn_env.resources_collected,
        md_dqn_env.resources_collected
    ]

    steps = [
        rule_env.steps,
        rl_env.steps,
        dqn_env.steps,
        md_dqn_env.steps
    ]

    rewards = [
        rule_env.total_reward,
        rl_env.total_reward,
        dqn_env.total_reward,
        md_dqn_env.total_reward
    ]

    collisions = [
        rule_env.collisions,
        rl_env.collisions,
        dqn_env.collisions,
        md_dqn_env.collisions
    ]

    print(f"{'Metric':<20} {'Rule-Based':>12} {'Q-Learning':>12} {'DQN':>12} {'MD-DQN':>12}")
    print("-" * 70)

    print(f"{'Resources':<20} {resources[0]:>12} {resources[1]:>12} {resources[2]:>12} {resources[3]:>12}")
    print(f"{'Collisions':<20} {collisions[0]:>12} {collisions[1]:>12} {collisions[2]:>12} {collisions[3]:>12}")
    print(f"{'Steps':<20} {steps[0]:>12} {steps[1]:>12} {steps[2]:>12} {steps[3]:>12}")
    print(f"{'Total Reward':<20} {round(rewards[0],2):>12} {round(rewards[1],2):>12} {round(rewards[2],2):>12} {round(rewards[3],2):>12}")

    # -----------------------------
    # 📈 LEARNING CURVE
    # -----------------------------
    plt.figure()
    plt.plot(smooth(rl_rewards), label="Q-Learning")
    plt.plot(smooth(dqn_rewards), label="DQN")
    plt.plot(smooth(md_dqn_rewards), label="MD-DQN")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # -----------------------------
    # 📊 BAR CHARTS
    # -----------------------------

    # Resources
    plt.figure()
    plt.bar(agents, resources)
    plt.title("Resources Collected")
    plt.show()

    # Steps
    plt.figure()
    plt.bar(agents, steps)
    plt.title("Steps Taken")
    plt.show()

    # Reward
    plt.figure()
    plt.bar(agents, rewards)
    plt.title("Total Reward")
    plt.show()

    # Collisions
    plt.figure()
    plt.bar(agents, collisions)
    plt.title("Collisions")
    plt.show()