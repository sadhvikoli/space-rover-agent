import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Feedforward neural network for DQN.
    Input: numeric state vector
    Output: Q-values for 4 actions
    """
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Experience replay buffer.
    """
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for the space rover project.
    """

    def __init__(self, state_dim=6, action_dim=4, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.alpha = 1e-3
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

        self.batch_size = 64
        self.target_update_freq = 20

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(capacity=20000)
        self.train_steps = 0

    def featurize_state(self, env):
        """
        Convert GridWorld state into a numeric feature vector.
        """
        x, y = env.agent_pos
        size = env.size

        resource_positions = list(zip(*np.where(env.grid == 1)))
        charger_positions = list(zip(*np.where(env.grid == 2)))

        def nearest_distance(positions):
            if not positions:
                return float(size * 2)
            return min(abs(px - x) + abs(py - y) for px, py in positions)

        nearest_resource = nearest_distance(resource_positions)
        nearest_charger = nearest_distance(charger_positions)
        remaining_resources = len(resource_positions)

        state_vec = np.array([
            x / (size - 1),
            y / (size - 1),
            env.energy / 100.0,
            remaining_resources / max(1, env.num_resources),
            nearest_resource / (2 * size),
            nearest_charger / (2 * size),
        ], dtype=np.float32)

        return state_vec

    def choose_action(self, state_vec, greedy=False):
        """
        Epsilon-greedy action selection.
        If greedy=True, always exploit.
        """
        if (not greedy) and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_weights(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)