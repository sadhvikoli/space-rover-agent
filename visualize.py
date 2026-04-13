import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from env import GridWorld
from agent import QAgent
from dqn_agent import DQNAgent
from md_dqn_agent import MDDQNAgent
from rule_agent import RuleBasedAgent
from train import train
from train_dqn import train_dqn
from train_md_dqn import train_md_dqn

# faster rendering
plt.rcParams['figure.dpi'] = 80

# color map for grid cells
# order: empty, obstacle, resource, charger, trail, rover
COLORS = ['white', 'dimgray', 'orange', 'limegreen', 'lightsteelblue', 'royalblue']
CMAP = ListedColormap(COLORS)

# cell values for rendering
EMPTY_VAL    = 0
OBSTACLE_VAL = 1
RESOURCE_VAL = 2
CHARGER_VAL  = 3
TRAIL_VAL    = 4
ROVER_VAL    = 5


def build_render_grid(env, trail):
    """
    Build a numeric grid for matplotlib to render.
    Maps env grid values to display values and overlays trail and rover.
    """
    render = np.zeros((env.size, env.size), dtype=int)

    for r in range(env.size):
        for c in range(env.size):
            val = env.grid[r][c]
            if val == -1:
                render[r][c] = OBSTACLE_VAL
            elif val == 1:
                render[r][c] = RESOURCE_VAL
            elif val == 2:
                render[r][c] = CHARGER_VAL
            else:
                render[r][c] = EMPTY_VAL

    # mark trail
    for (tr, tc) in trail:
        if render[tr][tc] == EMPTY_VAL:
            render[tr][tc] = TRAIL_VAL

    # mark rover position
    rx, ry = env.agent_pos
    render[rx][ry] = ROVER_VAL

    return render


def visualize_agent(agent_name, env, get_action_fn, delay=0.05):
    """
    Visualize one agent running on the grid step by step.

    Args:
        agent_name   : string label shown in title
        env          : GridWorld instance
        get_action_fn: function that returns next action
        delay        : seconds between each step
    """
    env.soft_reset()
    trail = set()

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.ion()
    plt.show()

    # create image once and update data instead of redrawing every step
    render = build_render_grid(env, trail)
    img = ax.imshow(render, cmap=CMAP, vmin=0, vmax=5)

    # draw grid lines once
    for x in range(env.size + 1):
        ax.axhline(x - 0.5, color='lightgray', linewidth=0.5)
        ax.axvline(x - 0.5, color='lightgray', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    # legend once
    legend = [
        Patch(color='royalblue',      label='Rover'),
        Patch(color='orange',         label='Resource'),
        Patch(color='limegreen',      label='Charger'),
        Patch(color='dimgray',        label='Obstacle'),
        Patch(color='lightsteelblue', label='Trail'),
    ]
    ax.legend(handles=legend, loc='upper center',
              bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8)

    while True:
        # mark current position as trail before moving
        trail.add(env.agent_pos)

        # get action from agent
        action = get_action_fn(env)

        # take step
        _, _, done = env.step(action)

        # update image data only — much faster than ax.clear() + ax.imshow()
        render = build_render_grid(env, trail)
        img.set_data(render)

        # update title with live stats
        ax.set_title(
            f"{agent_name}\n"
            f"Step: {env.steps}  |  Energy: {env.energy}  |  "
            f"Resources: {env.resources_collected}  |  Collisions: {env.collisions}",
            fontsize=11
        )

        # fast redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(delay)

        if done or env.steps >= 200:
            # show final state for 2 seconds
            time.sleep(2)
            break

    plt.close()


if __name__ == "__main__":

    # ── Train all agents ──────────────────────────────────────────────────────
    print("Training Q-Learning agent...")
    rl_agent = QAgent()
    rl_env = train(rl_agent, episodes=100)

    print("Training DQN agent...")
    dqn_agent = DQNAgent()
    dqn_env = train_dqn(dqn_agent, episodes=100)
    dqn_agent.save_weights("source_dqn.pth")

    print("Training MD-DQN agent...")
    md_agent = MDDQNAgent()
    md_agent.load_source_weights("source_dqn.pth")
    md_env = train_md_dqn(md_agent, episodes=100)

    print("Done training. Starting visualization...\n")

    # ── Rule-Based Agent ──────────────────────────────────────────────────────
    rule_env = GridWorld()
    rule_agent = RuleBasedAgent()
    visualize_agent(
        agent_name="Rule-Based Agent",
        env=rule_env,
        get_action_fn=lambda env: rule_agent.select_action(env),
        delay=0.05
    )

    # ── Q-Learning Agent ──────────────────────────────────────────────────────
    visualize_agent(
        agent_name="Q-Learning Agent",
        env=rl_env,
        get_action_fn=lambda env: rl_agent.choose_action(env.get_state()),
        delay=0.05
    )

    # ── DQN Agent ─────────────────────────────────────────────────────────────
    visualize_agent(
        agent_name="DQN Agent",
        env=dqn_env,
        get_action_fn=lambda env: dqn_agent.choose_action(
            dqn_agent.featurize_state(env), greedy=True
        ),
        delay=0.05
    )

    # ── MD-DQN Agent ──────────────────────────────────────────────────────────
    visualize_agent(
        agent_name="MD-DQN Agent",
        env=md_env,
        get_action_fn=lambda env: md_agent.choose_action(
            md_agent.featurize_state(env), greedy=True
        ),
        delay=0.05
    )

    print("Visualization complete!")