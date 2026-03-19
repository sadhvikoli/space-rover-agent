import random


class QAgent:
    """
    Q-learning agent for simulated planetary exploration.

    This agent learns by trial and error — it tries actions, receives rewards,
    and updates a Q-table to remember which actions work best in each state.
    Over time it shifts from random exploration to intelligent exploitation.

    Q-table structure:
        { (x, y, energy_bucket) : [Q_up, Q_down, Q_left, Q_right] }

    Q-learning update formula:
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s')) - Q(s, a))

    Where:
        alpha (learning rate) : how much to update Q values each step
        gamma (discount)      : how much future rewards are worth vs immediate
        epsilon               : probability of taking a random action (exploration)
    """

    def __init__(self):
        """
        Initialize the Q-learning agent with hyperparameters.

        Hyperparameters:
            alpha         : learning rate — how fast Q values update (0.1)
            gamma         : discount factor — how much future reward matters (0.9)
            epsilon       : starting exploration rate — 1.0 means fully random
            epsilon_min   : minimum exploration rate, never goes below this
            epsilon_decay : multiply epsilon by this after each episode to reduce exploration
            q_table       : dictionary storing Q values for each (state, action) pair
            actions       : number of possible actions (4 — UP, DOWN, LEFT, RIGHT)
        """
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0        # start at 1.0 — fully random exploration at first
        self.epsilon_min = 0.1    # never go below 10% random — always explore a little
        self.epsilon_decay = 0.95 # multiply by 0.95 each episode to gradually reduce exploration
        self.q_table = {}         # empty table — agent knows nothing at the start
        self.actions = 4          # UP, DOWN, LEFT, RIGHT

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.

        With probability epsilon  -> pick a random action (explore)
        With probability 1-epsilon -> pick the action with highest Q value (exploit)

        Early in training epsilon is high so the agent explores a lot.
        Over time epsilon decays so the agent exploits what it has learned.

        Args:
            state (tuple): current state (x, y, energy_bucket)

        Returns:
            action (int): chosen action — one of UP, DOWN, LEFT, RIGHT (0-3)
        """
        # explore — take a random action
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)

        # exploit — pick the action with the highest Q value for this state
        # if state hasn't been seen before, default all Q values to 0
        q_values = self.q_table.get(state, [0] * self.actions)
        return q_values.index(max(q_values))

    def update(self, state, action, reward, next_state):
        """
        Update the Q-table after taking an action.

        Applies the Q-learning formula to adjust the Q value for the
        (state, action) pair based on the reward received and the best
        possible future reward from the next state.

        Args:
            state      (tuple): state before the action (x, y, energy_bucket)
            action     (int)  : action that was taken
            reward     (float): reward received from the environment
            next_state (tuple): state after the action
        """
        # initialize Q values to 0 if these states haven't been seen before
        self.q_table.setdefault(state, [0] * self.actions)
        self.q_table.setdefault(next_state, [0] * self.actions)

        # get current Q value for this (state, action) pair
        current_q = self.q_table[state][action]

        # get the best Q value available from the next state
        best_next_q = max(self.q_table[next_state])

        # apply Q-learning formula to update the Q value
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)

    def decay_epsilon(self):
        """
        Reduce epsilon after each training episode.

        Multiplies epsilon by epsilon_decay (0.95) so the agent gradually
        shifts from exploration to exploitation as it learns the environment.
        Stops decaying once epsilon reaches epsilon_min (0.1).
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay