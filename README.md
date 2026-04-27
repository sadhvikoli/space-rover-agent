# Space Rover Agent — Autonomous Planetary Exploration

## Project Overview

This project simulates an autonomous rover navigating a 2D grid-based planetary surface. The rover must collect resources, avoid obstacles, and manage energy by visiting charging stations — all without human intervention.

We explore and compare four approaches:

- Rule-Based Agent — fixed logic, no learning
- Q-Learning Agent — tabular reinforcement learning
- Deep Q-Network (DQN) — neural network-based learning
- MD-DQN (Transfer Learning) — improves DQN using prior knowledge

---

## Project Structure

```
space-rover-agent/
├── env.py              # Grid environment (state, rewards, transitions)
├── rule_agent.py       # Rule-based baseline agent
├── agent.py            # Q-learning implementation
├── dqn_agent.py        # Deep Q-Network agent
├── md_dqn_agent.py     # Transfer learning (MD-DQN)
├── train.py            # Training loops
├── main.py             # Runs experiments & comparisons
├── analysis.py         # Generates plots and metrics
└── models/             # Saved trained models (.pth files)
```

---

## How to Run

**Install dependencies**
```bash
pip install numpy torch matplotlib
```

**Run full experiment**
```bash
python main.py
```

**Run analysis (plots)**
```bash
python analysis.py
```

---

## Environment

- Grid size: 8x8
- Randomized maps across runs
- Rover starts at (0, 0)

| Cell      | Symbol | Description       |
|-----------|--------|-------------------|
| Empty     | .      | Free space        |
| Obstacle  | #      | Blocked cell      |
| Resource  | R      | +10 reward        |
| Charger   | C      | Restores energy   |
| Rover     | V      | Agent position    |

---

## Reward Design

| Event            | Reward |
|------------------|--------|
| Collect resource | +10    |
| Move step        | -0.1   |
| Collision        | -5     |
| Recharge         | +5     |

---

## Agents

### 1. Rule-Based Agent

Uses predefined heuristics with no learning capability. Serves as the baseline for comparison.

### 2. Q-Learning Agent

Tabular reinforcement learning that learns Q-values for each state-action pair.

**State Representation:** `(row, col, energy_level)`

| Parameter          | Value      |
|--------------------|------------|
| Learning rate (a)  | 0.1        |
| Discount factor (y)| 0.9        |
| Exploration (e)    | 1.0 -> 0.1 |
| Episodes           | 100        |

### 3. Deep Q-Network (DQN)

Uses a neural network to approximate Q-values, enabling it to handle larger state spaces.

**Architecture:**
- Input: state feature vector
- Hidden layers: 128 -> 128 (ReLU)
- Output: Q-values for 4 actions

**Key Components:** Replay Buffer, Target Network, epsilon-greedy policy

| Parameter     | Value      |
|---------------|------------|
| Learning rate | 0.001      |
| Discount (y)  | 0.99       |
| Exploration   | 1.0 -> 0.05|
| Batch size    | 64         |
| Target update | 20         |

### 4. MD-DQN (Transfer Learning)

Extends DQN by incorporating a pretrained source model, combining learned prior knowledge with new experience.

**Modified Target:**
```
y = max(r + gamma * max Q_target(s', a'), Q_source(s, a))
```

**Benefits:** Faster learning, better early performance  
**Limitation:** Sensitive to source-target similarity

---

## Results Summary

| Metric        | Rule-Based | Q-Learning | DQN   | MD-DQN |
|---------------|------------|------------|-------|--------|
| Resources     | 3.4        | 3.4        | 3.2   | 3.8    |
| Collisions    | 0.0        | 2.8        | 0.0   | 0.0    |
| Steps         | 180.0      | 160.0      | 140.8 | 160.0  |
| Total Reward  | -2.0       | -26.0      | 3.84  | 6.0    |

**Key Findings:**
- MD-DQN achieved the highest rewards, best resource collection, and fastest convergence
- DQN was the most efficient in terms of steps taken with stable learning
- Q-Learning showed slower and less stable performance
- Rule-Based was safe but inefficient overall

---

## Key Concepts

- Reinforcement Learning
- Bellman Equation
- Exploration vs Exploitation
- Experience Replay
- Transfer Learning

---

## Future Work

- Multi-source transfer learning
- More complex environments
- Continuous state spaces
- Advanced architectures (Double DQN, PPO)

---

## Conclusion

MD-DQN improves learning by leveraging prior knowledge, resulting in faster convergence and better overall performance. While DQN is more efficient in navigation, MD-DQN achieves higher rewards, demonstrating the effectiveness of transfer learning in reinforcement learning.

---

## GitHub

https://github.com/sadhvikoli/space-rover-agent
