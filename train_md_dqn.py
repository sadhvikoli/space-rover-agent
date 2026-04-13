from env import GridWorld


def train_md_dqn(agent, episodes=300):
    """
    Train MD-DQN agent on a target task.
    """
    env = GridWorld()

    for ep in range(episodes):
        env.soft_reset()
        state = agent.featurize_state(env)

        total_reward = 0.0
        total_loss = 0.0
        updates = 0

        while True:
            action = agent.choose_action(state)
            _, reward, done = env.step(action)
            next_state = agent.featurize_state(env)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()

            if loss is not None:
                total_loss += loss
                updates += 1

            state = next_state
            total_reward += reward

            if done or env.steps >= 200:
                break

        agent.decay_epsilon()

        if (ep + 1) % agent.target_update_freq == 0:
            agent.update_target_network()

        if (ep + 1) % 10 == 0:
            avg_loss = total_loss / updates if updates > 0 else 0.0
            print(
                f"MD-DQN Episode {ep + 1} | Reward: {round(total_reward, 2)} "
                f"| Resources: {env.resources_collected} "
                f"| Epsilon: {round(agent.epsilon, 3)} "
                f"| Avg Loss: {round(avg_loss, 4)}"
            )

    return env