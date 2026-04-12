from env import GridWorld


def train(agent, episodes=300, env=None):
    """
    Train tabular Q-learning on a fixed map.
    """
    if env is None:
        env = GridWorld()

    for ep in range(episodes):
        env.soft_reset()
        state = env.get_state()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done or env.steps >= 200:
                break

        agent.decay_epsilon()

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep + 1} | Reward: {round(total_reward, 2)} "
                f"| Resources: {env.resources_collected} "
                f"| Epsilon: {round(agent.epsilon, 3)}"
            )

    return env