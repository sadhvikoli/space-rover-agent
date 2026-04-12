import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Neural network for Q-value prediction.
    Input: state feature vector
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
    Standard replay buffer for off-policy training.
    Stores: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity=10000):
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


class MDDQNAgent:
    """
    ModelDiff-inspired DQN agent.

    Main idea:
    - train a source DQN first on a source task
    - freeze that source network
    - when training on target task, use:
          target = max(standard_dqn_target, source_lower_bound)

    This is an approximate student-friendly version of MD-DQN.

    Notes:
    - lower_bound comes from the pretrained source network
    - lower_bound_scale can reduce overly aggressive transfer
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

        # scale source lower bound a bit for safety
        self.lower_bound_scale = 0.85

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # target-task learner
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # frozen source model (loaded later)
        self.source_net = QNetwork(state_dim, action_dim).to(self.device)
        self.source_loaded = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(capacity=10000)
        self.train_steps = 0

    def featurize_state(self, env):
        """
        Convert GridWorld environment into numeric features.

        Features:
            0. normalized row
            1. normalized col
            2. normalized energy
            3. normalized remaining resources
            4. normalized nearest resource distance
            5. normalized nearest charger distance
        """
        x, y = env.agent_pos
        size = env.size

        resource_positions = list(zip(*np.where(env.grid == 1)))  # RESOURCE
        charger_positions = list(zip(*np.where(env.grid == 2)))   # CHARGER

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
        if (not greedy) and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def load_source_weights(self, source_path):
        """
        Load pretrained source DQN weights into source_net and freeze it.
        """
        state_dict = torch.load(source_path, map_location=self.device)
        self.source_net.load_state_dict(state_dict)
        self.source_net.eval()

        for param in self.source_net.parameters():
            param.requires_grad = False

        self.source_loaded = True

    def save_target_weights(self, target_path):
        """
        Save the learned target-task network.
        """
        torch.save(self.q_net.state_dict(), target_path)

    def update(self):
        """
        One MD-DQN update step.

        Standard DQN target:
            r + gamma * max_a' Q_target(next_state, a')

        MD-DQN-inspired target:
            max(standard_target, source_lower_bound)

        source_lower_bound is approximated using the pretrained source network.
        """
        if len(self.memory) < self.batch_size:
            return None

        if not self.source_loaded:
            raise RuntimeError("Source network not loaded. Call load_source_weights() first.")

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # current Q(s,a)
        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            # standard DQN target
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            standard_target = rewards + (1 - dones) * self.gamma * max_next_q

            # ModelDiff-inspired lower bound from source policy/value
            source_q_values = self.source_net(states)
            source_lower_bound = source_q_values.gather(1, actions) * self.lower_bound_scale

            # MD-DQN target
            target_q = torch.maximum(standard_target, source_lower_bound)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)