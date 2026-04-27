import numpy as np
import random
from env import UP, DOWN, LEFT, RIGHT, RESOURCE, CHARGER, OBSTACLE

class RuleBasedAgent:
    """
    Rule-based rover agent for simulated planetary exploration.

    This agent does not learn. It follows fixed heuristics:
    - prioritize survival (energy)
    - then resource collection
    - otherwise move randomly

    Acts as a baseline for comparison with learning agents.
    """

    def __init__(self, low_energy_threshold=25):
        """
        Initialize rule-based agent.

        Args:
            low_energy_threshold (int): energy level below which the agent
                                        prioritizes charging.
        """
        self.low_energy_threshold = low_energy_threshold

    def select_action(self, env):
        """
        Decide next action based on current environment state.
        """
        x, y = env.agent_pos

        resources = list(zip(*np.where(env.grid == RESOURCE)))
        chargers = list(zip(*np.where(env.grid == CHARGER)))

        # priority 1: recharge if energy is low
        if env.energy < self.low_energy_threshold and chargers:
            target = self._nearest((x, y), chargers)
            return self._move_toward(env, target)

        # priority 2: collect nearest resource
        if resources:
            target = self._nearest((x, y), resources)
            return self._move_toward(env, target)

        # fallback: random movement
        return self._random_valid_move(env)

    def _nearest(self, pos, targets):
        """
        Return nearest target using Manhattan distance.
        """
        return min(
            targets,
            key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1])
        )

    def _move_toward(self, env, target):
        """
        Move one step toward target while avoiding obstacles.
        """
        x, y = env.agent_pos
        tx, ty = target

        possible_moves = []

        if tx < x:
            possible_moves.append(UP)
        elif tx > x:
            possible_moves.append(DOWN)

        if ty < y:
            possible_moves.append(LEFT)
        elif ty > y:
            possible_moves.append(RIGHT)

        for a in [UP, DOWN, LEFT, RIGHT]:
            if a not in possible_moves:
                possible_moves.append(a)

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

        return self._random_valid_move(env)

    def _random_valid_move(self, env):
        """
        Return a random valid move avoiding obstacles.
        """
        x, y = env.agent_pos
        moves = []

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

        return random.choice(moves) if moves else random.choice(
            [UP, DOWN, LEFT, RIGHT]
        )
