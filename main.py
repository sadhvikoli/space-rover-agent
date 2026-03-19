from env import GridWorld
from agent import QAgent
from rule_agent import RuleBasedAgent
from train import train


if __name__ == "__main__":
    """
    Main entry point for the Space Rover simulation.

    Runs two agents on the same type of environment and compares their performance:

        1. Rule-Based Agent — follows fixed hand-coded rules, no learning
        2. Q-Learning Agent — learns through trial and error over 100 episodes

    Comparison metrics:
        - Resources collected : how many of the 4 resources were picked up
        - Collisions          : how many times the rover hit a wall or obstacle
        - Steps taken         : total number of moves made
        - Total reward        : cumulative reward across the episode
    """

    # Rule-Based Agent 
    # runs on a fresh random map with no prior learning
    print("Running Rule-Based Agent...")
    rule_env = GridWorld()
    rule_agent = RuleBasedAgent()
    state = rule_env.get_state()

    while True:
        # agent picks action based on rules — no learning involved
        action = rule_agent.select_action(rule_env)
        next_state, reward, done = rule_env.step(action)
        state = next_state

        # stop if energy runs out or step limit is reached
        if done or rule_env.steps >= 200:
            break

    print(f"Rule-Based | Resources: {rule_env.resources_collected} | "
          f"Collisions: {rule_env.collisions} | Steps: {rule_env.steps}")

    # RL Agent — Training Phase─
    # trains on a fixed map for 100 episodes, building up Q-table knowledge
    print("\nTraining RL Agent...")
    rl_agent = QAgent()

    # train() returns the same environment it trained on so we can reuse the map
    rl_env = train(rl_agent, episodes=100)

    # RL Agent — Final Episode
    # run the trained agent on the same map it learned — epsilon is now low
    # so it mostly exploits its learned Q-table instead of exploring randomly
    print("\nRunning RL Agent final episode...")
    rl_env.soft_reset()  # reset rover position but keep the same map
    state = rl_env.get_state()

    while True:
        # agent picks action based on learned Q-table
        action = rl_agent.choose_action(state)
        next_state, reward, done = rl_env.step(action)
        state = next_state

        # stop if energy runs out or step limit is reached
        if done or rl_env.steps >= 200:
            break

    print(f"RL Agent   | Resources: {rl_env.resources_collected} | "
          f"Collisions: {rl_env.collisions} | Steps: {rl_env.steps}")

    # Comparison Table 
    # side by side comparison of both agents across all metrics
    print("\n-*-*- COMPARISON -*-*-")
    print(f"{'Metric':<20} {'Rule-Based':>12} {'RL Agent':>12}")
    print("-" * 46)
    print(f"{'Resources':<20} {rule_env.resources_collected:>12} {rl_env.resources_collected:>12}")
    print(f"{'Collisions':<20} {rule_env.collisions:>12} {rl_env.collisions:>12}")
    print(f"{'Steps':<20} {rule_env.steps:>12} {rl_env.steps:>12}")
    print(f"{'Total Reward':<20} {round(rule_env.total_reward,2):>12} {round(rl_env.total_reward,2):>12}")