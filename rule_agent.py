import numpy as np
import random
from env import UP, DOWN, LEFT, RIGHT, RESOURCE, CHARGER, OBSTACLE


class RuleBasedAgent:
    """
    Rule-based rover agent for simulated planetary exploration.

    This agent does not learn — it follows a fixed set of hand-coded rules
    to decide what action to take at each step. It serves as a baseline
    to compare against the Q-learning agent.

    Strategy (in priority order):
        1. If energy is low -> navigate to the nearest charging station
        2. Otherwise -> navigate to the nearest resource and collect it
        3. If nothing found -> take a random valid move
    """

    def __init__(self, low_energy_threshold=25):
        """
        Initialize the rule-based agent.

        Args:
            low_energy_threshold (int): energy level below which the agent
                                        prioritizes recharging over collecting.
                                        Default is 25.
        """
        self.low_energy_threshold = low_energy_threshold

    def select_action(self, env):
        """
        Select the next action based on the current environment state.

        Scans the grid to find all remaining resources and chargers,
        then applies the priority rules to pick the best action.

        Args:
            env (GridWorld): the current environment instance

        Returns:
            action (int): one of UP, DOWN, LEFT, RIGHT
        """
        x, y = env.agent_pos

        # scan grid to find all resource and charger positions
        resources = list(zip(*np.where(env.grid == RESOURCE)))
        chargers = list(zip(*np.where(env.grid == CHARGER)))

        # rule 1 — if energy is critically low, go recharge first
        if env.energy < self.low_energy_threshold and chargers:
            target = self._nearest((x, y), chargers)
            return self._move_toward(env, target)

        # rule 2 — move toward the nearest uncollected resource
        if resources:
            target = self._nearest((x, y), resources)
            return self._move_toward(env, target)

        # rule 3 — nothing to do, take a random valid move
        return self._random_valid_move(env)

    def _nearest(self, pos, targets):
        """
        Find the nearest target position using Manhattan distance.

        Manhattan distance = |row1 - row2| + |col1 - col2|
        This is the correct distance metric for grid movement
        since the rover cannot move diagonally.

        Args:
            pos     (tuple): current position (x, y)
            targets (list) : list of target positions [(x1,y1), (x2,y2), ...]

        Returns:
            tuple: the closest target position
        """
        return min(targets, key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1]))

    def _move_toward(self, env, target):
        """
        Move one step toward the target, avoiding walls and obstacles.

        First tries to move directly toward the target (preferred direction).
        If that's blocked, tries all other directions as backup so the
        rover doesn't get stuck behind obstacles.

        Args:
            env    (GridWorld): current environment instance
            target (tuple)    : target position (x, y) to move toward

        Returns:
            action (int): best valid action toward the target
        """
        x, y = env.agent_pos
        tx, ty = target

        possible_moves = []

        # add preferred directions toward the target first
        if tx < x:
            possible_moves.append(UP)
        elif tx > x:
            possible_moves.append(DOWN)

        if ty < y:
            possible_moves.append(LEFT)
        elif ty > y:
            possible_moves.append(RIGHT)

        # add remaining directions as backup to avoid getting stuck
        for a in [UP, DOWN, LEFT, RIGHT]:
            if a not in possible_moves:
                possible_moves.append(a)

        # pick the first move that doesn't hit a wall or obstacle
        for action in possible_moves:
            nx, ny = x, y
            if action == UP:
                nx -= 1
            elif action == DOWN:
                nx += 1
            elif action == LEFT:
                ny -= 1
            elif action == RIGHT:
                ny += 1

            if 0 <= nx < env.size and 0 <= ny < env.size:
                if env.grid[nx][ny] != OBSTACLE:
                    return action

        # completely blocked in all directions — fallback to random
        return self._random_valid_move(env)

    def _random_valid_move(self, env):
        """
        Pick a random valid move that doesn't hit a wall or obstacle.

        Used as a fallback when the rover is stuck or has no target.
        If all moves are blocked (very rare), picks a random direction anyway.

        Args:
            env (GridWorld): current environment instance

        Returns:
            action (int): a random valid action
        """
        x, y = env.agent_pos
        moves = []

        # check all 4 directions and keep the ones that are valid
        for action in [UP, DOWN, LEFT, RIGHT]:
            nx, ny = x, y
            if action == UP:
                nx -= 1
            elif action == DOWN:
                nx += 1
            elif action == LEFT:
                ny -= 1
            elif action == RIGHT:
                ny += 1

            if 0 <= nx < env.size and 0 <= ny < env.size:
                if env.grid[nx][ny] != OBSTACLE:
                    moves.append(action)

        return random.choice(moves) if moves else random.choice([UP, DOWN, LEFT, RIGHT])