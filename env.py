import numpy as np
import random

# Cell type constants — represent what's in each grid cell
# Using named constants makes the code readable — OBSTACLE is clearer than -1
EMPTY = 0
OBSTACLE = -1
RESOURCE = 1
CHARGER = 2

# Action constants — represent the 4 directions the rover can move
# Using names like UP instead of 0 makes the agent code much easier to read
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class GridWorld:
    """
    2D grid-based environment simulating a planetary surface.

    The grid contains obstacles, collectible resources, and charging stations.
    The rover starts at (0, 0) and must collect resources while managing energy.

    Cell types:
        EMPTY    ( 0) — free cell the rover can move into
        OBSTACLE (-1) — blocked cell, rover cannot enter
        RESOURCE ( 1) — collectible item, gives +10 reward when picked up
        CHARGER  ( 2) — charging station, restores +20 energy when visited

    Rewards:
        +10   for collecting a resource
        -0.1  for each valid move (encourages efficiency)
        -10    for hitting a wall or obstacle (discourages collisions)
    """

    def __init__(self, size=8, num_obstacles=8, num_resources=4, num_chargers=2):
        """
        Initialize the environment with grid dimensions and item counts.

        Args:
            size         : width and height of the square grid (default 8)
            num_obstacles: number of obstacle cells to place (default 8)
            num_resources: number of resources to place (default 4)
            num_chargers : number of charging stations to place (default 2)
        """
        self.size = size
        self.num_obstacles = num_obstacles
        self.num_resources = num_resources
        self.num_chargers = num_chargers
        self.reset()

    def reset(self):
        """
        Fully reset the environment — new grid, new map, new rover position.

        Creates a blank grid, places all items randomly, and resets all
        counters. Saves the generated map so soft_reset() can restore it.

        Returns:
            state (tuple): initial state (x, y, energy_bucket)
        """
        # create a blank grid filled with zeros (EMPTY)
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.agent_pos = (0, 0)
        self.energy = 100
        self.resources_collected = 0
        self.steps = 0
        self.collisions = 0
        self.total_reward = 0.0

        # place items randomly on the grid
        self._place(OBSTACLE, self.num_obstacles)
        self._place(RESOURCE, self.num_resources)
        self._place(CHARGER, self.num_chargers)

        # save a copy of the generated map so soft_reset() can restore it
        self.original_grid = self.grid.copy()

        return self.get_state()

    def soft_reset(self):
        """
        Reset the rover and stats only — keep the exact same map layout.

        Used during RL training so the agent learns the same map across
        multiple episodes instead of starting from scratch each time.

        Returns:
            state (tuple): initial state (x, y, energy_bucket)
        """
        self.grid = self.original_grid.copy()  # restore original map
        self.agent_pos = (0, 0)
        self.energy = 100
        self.resources_collected = 0
        self.steps = 0
        self.collisions = 0
        self.total_reward = 0.0
        return self.get_state()

    def _place(self, value, count):
        """
        Randomly place a given number of items on the grid.

        Keeps trying random positions until it finds an empty cell
        that isn't the rover's starting position (0, 0).

        Args:
            value (int): cell type to place (OBSTACLE, RESOURCE, or CHARGER)
            count (int): how many to place
        """
        placed = 0
        while placed < count:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if self.grid[x][y] == EMPTY and (x, y) != self.agent_pos:
                self.grid[x][y] = value
                placed += 1

    def get_state(self):
        """
        Return the current state as a hashable tuple for the Q-table.

        State = (row, col, energy_bucket)
        Energy is bucketed into 0-10 by dividing by 10 to reduce the
        number of unique states the Q-table needs to store.

        Returns:
            tuple: (x, y, energy // 10)
        """
        x, y = self.agent_pos
        return (x, y, self.energy // 10)

    def step(self, action):
        """
        Execute one action and return the result.

        Moves the rover in the given direction, applies rewards/penalties,
        handles resource collection and recharging, and checks if the
        episode is over.

        Args:
            action (int): one of UP, DOWN, LEFT, RIGHT

        Returns:
            state  (tuple) : new state after the action
            reward (float) : reward received for this action
            done   (bool)  : True if the episode is over (energy ran out)
        """
        x, y = self.agent_pos

        # calculate new position based on action
        if action == UP:
            x -= 1
        elif action == DOWN:
            x += 1
        elif action == LEFT:
            y -= 1
        elif action == RIGHT:
            y += 1

        # check if move hits a wall or obstacle — don't move, penalize
        if not (0 <= x < self.size and 0 <= y < self.size) or self.grid[x][y] == OBSTACLE:
            self.collisions += 1
            reward = -10
            self.total_reward += reward
            return self.get_state(), reward, False

        # valid move — update rover position and drain energy
        self.agent_pos = (x, y)
        self.energy -= 1
        self.steps += 1
        reward = -0.2  # small penalty per move to encourage efficiency

        # collect resource if rover lands on one
        if self.grid[x][y] == RESOURCE:
            reward += 10
            self.resources_collected += 1
            self.grid[x][y] = EMPTY  # remove resource from grid

        # recharge energy if rover is on a charging station
        if self.grid[x][y] == CHARGER:
            self.energy = min(100, self.energy + 20)  # cap at 100

        # episode ends when energy runs out
        done = self.energy <= 0
        self.total_reward += reward
        return self.get_state(), reward, done

    def render(self):
        """
        Print the current grid state to the terminal.

        Symbols:
            V — rover
            R — resource
            C — charging station
            # — obstacle
            . — empty cell
        """
        symbols = {EMPTY: '.', OBSTACLE: '#', RESOURCE: 'R', CHARGER: 'C'}
        print(f"\nStep: {self.steps} | Energy: {self.energy} | "
              f"Collected: {self.resources_collected} | Collisions: {self.collisions}")
        print('+' + '---' * self.size + '+')
        for r in range(self.size):
            row = '|'
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row += ' V '
                else:
                    row += f' {symbols[self.grid[r][c]]} '
            print(row + '|')
        print('+' + '---' * self.size + '+')