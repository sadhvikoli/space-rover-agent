import random

class QAgent:
    """
    Q-learning agent for simulated planetary exploration.

    The agent learns through interaction:
    - explores actions randomly at first
    - gradually exploits learned Q-values over time

    Q-table format:
        { (x, y, energy_bucket): [Q_up, Q_down, Q_left, Q_right] }

    Q-learning update rule:
        Q(s, a) ← Q(s, a) + α [r + γ max(Q(s')) − Q(s, a)]
    """

    def __init__(self):
        """
        Initialize Q-learning agent hyperparameters.
        """
        self.alpha = 0.1
        self.gamma = 0.9

        # exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99

        self.q_table = {}
        self.actions = 4  # UP, DOWN, LEFT, RIGHT

    def choose_action(self, state):
        """
        Select action using epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)

        q_values = self.q_table.get(state, [0] * self.actions)
        return q_values.index(max(q_values))

    def update(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule.
        """
        self.q_table.setdefault(state, [0] * self.actions)
        self.q_table.setdefault(next_state, [0] * self.actions)

        current_q = self.q_table[state][action]
        best_next_q = max(self.q_table[next_state])

        self.q_table[state][action] = (
            current_q +
            self.alpha * (reward + self.gamma * best_next_q - current_q)
        )

    def decay_epsilon(self):
        """
        Gradually reduce exploration rate after each episode.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
