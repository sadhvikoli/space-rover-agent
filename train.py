from env import GridWorld


def train(agent, episodes=100):
    """
    Train the Q-learning agent over multiple episodes on a fixed map.

    Creates one GridWorld map and reuses it across all episodes using
    soft_reset() so the agent learns the same environment repeatedly.
    This is important because Q-learning needs repeated exposure to the
    same states to build up accurate Q values.

    After each episode, epsilon is decayed so the agent gradually shifts
    from random exploration to exploiting what it has learned.

    Progress is printed every 10 episodes showing reward and resources
    collected so you can see the agent improving over time.

    Args:
        agent    (QAgent): the Q-learning agent to train
        episodes (int)   : number of training episodes to run (default 100)

    Returns:
        env (GridWorld): the environment the agent was trained on,
                         so main.py can reuse the same map for the final episode
    """
    # create one fixed map — agent will train on this same map every episode
    env = GridWorld()

    for ep in range(episodes):
        # reset rover position and stats but keep the same map layout
        env.soft_reset()
        state = env.get_state()
        total_reward = 0

        while True:
            # agent picks an action based on current state
            action = agent.choose_action(state)

            # environment executes the action and returns result
            next_state, reward, done = env.step(action)

            # agent updates Q-table based on what just happened
            agent.update(state, action, reward, next_state)

            # move to next state
            state = next_state
            total_reward += reward

            # stop episode if energy ran out or step limit reached
            # step limit prevents infinite loops if agent gets stuck
            if done or env.steps >= 200:
                break

        # reduce epsilon after each episode — less exploration, more exploitation
        agent.decay_epsilon()

        # print progress every 10 episodes so we can track learning
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Reward: {round(total_reward, 2)} | Resources: {env.resources_collected}")

    # return the trained environment so main.py can run a final episode on same map
    return env