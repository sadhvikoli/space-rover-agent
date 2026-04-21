from env import GridWorld
from agent import QAgent
from rule_agent import RuleBasedAgent
from train import train

from dqn_agent import DQNAgent
from train_dqn import train_dqn

from md_dqn_agent import MDDQNAgent
from train_md_dqn import train_md_dqn


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

    # overwrite the randomly generated map with the shared base map
    env.grid = base_grid.copy()
    env.original_grid = base_grid.copy()

    # reset rover state and stats
    env.agent_pos = (0, 0)
    env.energy = 100
    env.resources_collected = 0
    env.steps = 0
    env.collisions = 0
    env.total_reward = 0.0

    return env


if __name__ == "__main__":
    """
    Main entry point for the Space Rover simulation.

    All agents use the SAME map layout for fair comparison,
    but each gets its own environment object so results remain independent.
    """

    print("Creating base map...")
    base_env = GridWorld()
    base_grid = base_env.original_grid.copy()

    # create separate environments with the same map
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
    # Q-Learning Agent
    # -----------------------------
    print("\nTraining Q-Learning Agent...")
    rl_agent = QAgent()
    train(rl_agent, episodes=300, env=rl_env)

    print("\nRunning Q-Learning Agent final episode...")
    rl_env.soft_reset()
    state = rl_env.get_state()

    while True:
        action = rl_agent.choose_action(state)
        state, _, done = rl_env.step(action)

        if done or rl_env.steps >= 200:
            break

    print(f"Q-Learning | Resources: {rl_env.resources_collected} | "
          f"Collisions: {rl_env.collisions} | Steps: {rl_env.steps}")

    # -----------------------------
    # DQN Agent
    # -----------------------------
    print("\nTraining DQN Agent...")
    dqn_agent = DQNAgent()
    train_dqn(dqn_agent, episodes=300, env=dqn_env)

    print("\nRunning DQN Agent final episode...")
    dqn_env.soft_reset()
    state = dqn_agent.featurize_state(dqn_env)

    while True:
        action = dqn_agent.choose_action(state, greedy=True)
        _, _, done = dqn_env.step(action)
        state = dqn_agent.featurize_state(dqn_env)

        if done or dqn_env.steps >= 200:
            break

    print(f"DQN        | Resources: {dqn_env.resources_collected} | "
          f"Collisions: {dqn_env.collisions} | Steps: {dqn_env.steps}")

    # save source DQN weights
    dqn_agent.save_weights("source_dqn.pth")

    # -----------------------------
    # MD-DQN Agent
    # -----------------------------
    print("\nTraining MD-DQN Agent...")
    md_dqn_agent = MDDQNAgent()
    md_dqn_agent.load_source_weights("source_dqn.pth")
    train_md_dqn(md_dqn_agent, episodes=300, env=md_dqn_env)

    print("\nRunning MD-DQN Agent final episode...")
    md_dqn_env.soft_reset()
    state = md_dqn_agent.featurize_state(md_dqn_env)

    while True:
        action = md_dqn_agent.choose_action(state, greedy=True)
        _, _, done = md_dqn_env.step(action)
        state = md_dqn_agent.featurize_state(md_dqn_env)

        if done or md_dqn_env.steps >= 200:
            break

    print(f"MD-DQN     | Resources: {md_dqn_env.resources_collected} | "
          f"Collisions: {md_dqn_env.collisions} | Steps: {md_dqn_env.steps}")

    # -----------------------------
    # FINAL COMPARISON
    # -----------------------------
    print("\n-*-*- FINAL COMPARISON -*-*-")
    print(f"{'Metric':<20} {'Rule-Based':>12} {'Q-Learning':>12} {'DQN':>12} {'MD-DQN':>12}")
    print("-" * 70)

    print(f"{'Resources':<20} "
          f"{rule_env.resources_collected:>12} "
          f"{rl_env.resources_collected:>12} "
          f"{dqn_env.resources_collected:>12} "
          f"{md_dqn_env.resources_collected:>12}")

    print(f"{'Collisions':<20} "
          f"{rule_env.collisions:>12} "
          f"{rl_env.collisions:>12} "
          f"{dqn_env.collisions:>12} "
          f"{md_dqn_env.collisions:>12}")

    print(f"{'Steps':<20} "
          f"{rule_env.steps:>12} "
          f"{rl_env.steps:>12} "
          f"{dqn_env.steps:>12} "
          f"{md_dqn_env.steps:>12}")

    print(f"{'Total Reward':<20} "
          f"{round(rule_env.total_reward, 2):>12} "
          f"{round(rl_env.total_reward, 2):>12} "
          f"{round(dqn_env.total_reward, 2):>12} "
          f"{round(md_dqn_env.total_reward, 2):>12}")