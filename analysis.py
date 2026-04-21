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
import random


def clone_env(base_env, base_grid):
    """
    Create a fresh environment object with the exact same map layout
    as the base environment.
    """
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
    """
    Smooth a curve using moving average.
    """
    x = np.array(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def evaluate_rule_agent(env):
    """
    Run the rule-based agent once on the given environment.
    Returns a dictionary of final metrics.
    """
    agent = RuleBasedAgent()

    env.soft_reset()

    while True:
        action = agent.select_action(env)
        _, _, done = env.step(action)

        if done or env.steps >= 200:
            break

    return {
        "resources": env.resources_collected,
        "collisions": env.collisions,
        "steps": env.steps,
        "reward": env.total_reward,
    }


def evaluate_q_learning(env, episodes=300):
    """
    Train and evaluate Q-learning on one map.
    """
    agent = QAgent()
    env, rewards_history = train(agent, episodes=episodes, env=env)

    env.soft_reset()
    state = env.get_state()

    while True:
        action = agent.choose_action(state)
        state, _, done = env.step(action)

        if done or env.steps >= 200:
            break

    metrics = {
        "resources": env.resources_collected,
        "collisions": env.collisions,
        "steps": env.steps,
        "reward": env.total_reward,
    }

    return rewards_history, metrics


def evaluate_dqn(env, episodes=300, weight_path="source_dqn.pth"):
    """
    Train and evaluate DQN on one map.
    """
    agent = DQNAgent()
    env, rewards_history = train_dqn(agent, episodes=episodes, env=env)

    env.soft_reset()
    state = agent.featurize_state(env)

    while True:
        action = agent.choose_action(state, greedy=True)
        _, _, done = env.step(action)
        state = agent.featurize_state(env)

        if done or env.steps >= 200:
            break

    agent.save_weights(weight_path)

    metrics = {
        "resources": env.resources_collected,
        "collisions": env.collisions,
        "steps": env.steps,
        "reward": env.total_reward,
    }

    return rewards_history, metrics


def evaluate_md_dqn(env, episodes=300, weight_path="source_dqn.pth"):
    """
    Train and evaluate MD-DQN on one map.
    """
    agent = MDDQNAgent()
    agent.load_source_weights(weight_path)
    env, rewards_history = train_md_dqn(agent, episodes=episodes, env=env)

    env.soft_reset()
    state = agent.featurize_state(env)

    while True:
        action = agent.choose_action(state, greedy=True)
        _, _, done = env.step(action)
        state = agent.featurize_state(env)

        if done or env.steps >= 200:
            break

    metrics = {
        "resources": env.resources_collected,
        "collisions": env.collisions,
        "steps": env.steps,
        "reward": env.total_reward,
    }

    return rewards_history, metrics


def plot_learning_curves_with_variance(all_rl, all_dqn, all_md):
    """
    Plot mean learning curves with standard deviation bands.
    """
    all_rl = np.array(all_rl)
    all_dqn = np.array(all_dqn)
    all_md = np.array(all_md)

    mean_rl = np.mean(all_rl, axis=0)
    std_rl = np.std(all_rl, axis=0)

    mean_dqn = np.mean(all_dqn, axis=0)
    std_dqn = np.std(all_dqn, axis=0)

    mean_md = np.mean(all_md, axis=0)
    std_md = np.std(all_md, axis=0)

    s_mean_rl = smooth(mean_rl)
    s_mean_dqn = smooth(mean_dqn)
    s_mean_md = smooth(mean_md)

    s_low_rl = smooth(mean_rl - std_rl)
    s_high_rl = smooth(mean_rl + std_rl)

    s_low_dqn = smooth(mean_dqn - std_dqn)
    s_high_dqn = smooth(mean_dqn + std_dqn)

    s_low_md = smooth(mean_md - std_md)
    s_high_md = smooth(mean_md + std_md)

    episodes = np.arange(len(s_mean_rl))

    plt.figure(figsize=(10, 6))

    plt.plot(episodes, s_mean_rl, label="Q-Learning")
    plt.fill_between(episodes, s_low_rl, s_high_rl, alpha=0.2)

    plt.plot(episodes, s_mean_dqn, label="DQN")
    plt.fill_between(episodes, s_low_dqn, s_high_dqn, alpha=0.2)

    plt.plot(episodes, s_mean_md, label="MD-DQN")
    plt.fill_between(episodes, s_low_md, s_high_md, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Average Total Reward")
    plt.title("Average Learning Curve Across Multiple Maps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bar_chart(title, agents, values, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(agents, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    NUM_MAPS = 5
    EPISODES = 300

    all_rl_rewards = []
    all_dqn_rewards = []
    all_md_dqn_rewards = []

    rule_results = []
    rl_results = []
    dqn_results = []
    md_dqn_results = []

    for run in range(NUM_MAPS):
        print(f"\n========== MAP {run + 1}/{NUM_MAPS} ==========")

        random.seed(run)
        np.random.seed(run)

        base_env = GridWorld()
        base_grid = base_env.original_grid.copy()

        rule_env = clone_env(base_env, base_grid)
        rl_env = clone_env(base_env, base_grid)
        dqn_env = clone_env(base_env, base_grid)
        md_dqn_env = clone_env(base_env, base_grid)

        print("Running Rule-Based Agent...")
        rule_metrics = evaluate_rule_agent(rule_env)
        rule_results.append(rule_metrics)

        print("Training Q-Learning Agent...")
        rl_rewards, rl_metrics = evaluate_q_learning(rl_env, episodes=EPISODES)
        all_rl_rewards.append(rl_rewards)
        rl_results.append(rl_metrics)

        print("Training DQN Agent...")
        dqn_rewards, dqn_metrics = evaluate_dqn(
            dqn_env,
            episodes=EPISODES,
            weight_path=f"source_dqn_run_{run}.pth"
        )
        all_dqn_rewards.append(dqn_rewards)
        dqn_results.append(dqn_metrics)

        print("Training MD-DQN Agent...")
        md_dqn_rewards, md_dqn_metrics = evaluate_md_dqn(
            md_dqn_env,
            episodes=EPISODES,
            weight_path=f"source_dqn_run_{run}.pth"
        )
        all_md_dqn_rewards.append(md_dqn_rewards)
        md_dqn_results.append(md_dqn_metrics)

    avg_resources = [
        np.mean([x["resources"] for x in rule_results]),
        np.mean([x["resources"] for x in rl_results]),
        np.mean([x["resources"] for x in dqn_results]),
        np.mean([x["resources"] for x in md_dqn_results]),
    ]

    avg_collisions = [
        np.mean([x["collisions"] for x in rule_results]),
        np.mean([x["collisions"] for x in rl_results]),
        np.mean([x["collisions"] for x in dqn_results]),
        np.mean([x["collisions"] for x in md_dqn_results]),
    ]

    avg_steps = [
        np.mean([x["steps"] for x in rule_results]),
        np.mean([x["steps"] for x in rl_results]),
        np.mean([x["steps"] for x in dqn_results]),
        np.mean([x["steps"] for x in md_dqn_results]),
    ]

    avg_rewards = [
        np.mean([x["reward"] for x in rule_results]),
        np.mean([x["reward"] for x in rl_results]),
        np.mean([x["reward"] for x in dqn_results]),
        np.mean([x["reward"] for x in md_dqn_results]),
    ]

    agents = ["Rule", "Q-Learning", "DQN", "MD-DQN"]

    print("\n\n-*-*- AVERAGE FINAL COMPARISON ACROSS MAPS -*-*-")
    print(f"{'Metric':<20} {'Rule-Based':>12} {'Q-Learning':>12} {'DQN':>12} {'MD-DQN':>12}")
    print("-" * 70)

    print(f"{'Resources':<20} "
          f"{round(avg_resources[0], 2):>12} "
          f"{round(avg_resources[1], 2):>12} "
          f"{round(avg_resources[2], 2):>12} "
          f"{round(avg_resources[3], 2):>12}")

    print(f"{'Collisions':<20} "
          f"{round(avg_collisions[0], 2):>12} "
          f"{round(avg_collisions[1], 2):>12} "
          f"{round(avg_collisions[2], 2):>12} "
          f"{round(avg_collisions[3], 2):>12}")

    print(f"{'Steps':<20} "
          f"{round(avg_steps[0], 2):>12} "
          f"{round(avg_steps[1], 2):>12} "
          f"{round(avg_steps[2], 2):>12} "
          f"{round(avg_steps[3], 2):>12}")

    print(f"{'Total Reward':<20} "
          f"{round(avg_rewards[0], 2):>12} "
          f"{round(avg_rewards[1], 2):>12} "
          f"{round(avg_rewards[2], 2):>12} "
          f"{round(avg_rewards[3], 2):>12}")

    plot_learning_curves_with_variance(all_rl_rewards, all_dqn_rewards, all_md_dqn_rewards)
    plot_bar_chart("Average Resources Collected", agents, avg_resources, "Resources")
    plot_bar_chart("Average Collisions", agents, avg_collisions, "Collisions")
    plot_bar_chart("Average Steps Taken", agents, avg_steps, "Steps")
    plot_bar_chart("Average Total Reward", agents, avg_rewards, "Reward")