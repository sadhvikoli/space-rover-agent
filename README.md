# Space Rover Agent — Autonomous Planetary Exploration

## Project Overview

This project simulates an autonomous rover navigating a 2D grid-based planetary surface. The rover must collect resources, avoid obstacles, and manage its energy by visiting charging stations — all without human intervention.

Two agents are implemented and compared:
- **Rule-Based Agent** — follows fixed hand-coded rules, no learning
- **Q-Learning Agent** — learns through trial and error using Reinforcement Learning

---

## Project Structure
```
space-rover-agent/
├── env.py          # Grid environment — map, rover, rewards
├── agent.py        # Q-learning agent
├── rule_agent.py   # Rule-based agent
├── train.py        # Training loop for RL agent
└── main.py         # Run both agents and compare results
```

---

## How to Run

**Install dependencies:**
```bash
pip install numpy
```

**Run the simulation:**
```bash
python main.py
```

---

## Environment

- 8x8 grid map
- Rover starts at position (0, 0)
- Items placed randomly each run

| Cell | Symbol | Description |
|------|--------|-------------|
| Empty | . | Free cell rover can move into |
| Obstacle | # | Blocked cell, rover cannot enter |
| Resource | R | Collectible item, gives +10 reward |
| Charger | C | Restores +20 energy when visited |
| Rover | V | Current rover position |

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Collect resource | +10 |
| Valid move | -0.1 |
| Hit wall or obstacle | -5 |

---

## Agents

### Rule-Based Agent
Follows a fixed priority strategy:
1. If energy < 25 → go to nearest charging station
2. Go to nearest resource
3. If nothing found → random valid move

No learning involved. Serves as a baseline for comparison.

### Q-Learning Agent
Uses Reinforcement Learning to learn an optimal policy through trial and error.

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Alpha (α) | 0.1 | Learning rate |
| Gamma (γ) | 0.9 | Discount factor |
| Epsilon (ε) | 1.0 → 0.1 | Exploration rate, decays over episodes |
| Episodes | 100 | Training episodes on fixed map |

**State representation:** `(row, col, energy // 10)`  
**Q-table:** `{ state : [Q_up, Q_down, Q_left, Q_right] }`

---

## Sample Output
```
Running Rule-Based Agent...
Rule-Based | Resources: 4 | Collisions: 0 | Steps: 395

Training RL Agent...
Episode 10 | Reward: -121.8 | Resources: 1
Episode 20 | Reward: -140.0 | Resources: 4
Episode 50 | Reward: -35.0  | Resources: 1
Episode 90 | Reward: 16.6   | Resources: 4
Episode 100 | Reward: -10.0 | Resources: 3

Running RL Agent final episode...
RL Agent | Resources: 4 | Collisions: 1 | Steps: 200

-*-*- COMPARISON -*-*-
Metric                  Rule-Based     RL Agent
----------------------------------------------
Resources                        4            4
Collisions                       0            1
Steps                          395          200
Total Reward                 -10.0         15.0
```

---

## Key Observations

- The RL agent starts with random behavior and improves over 100 episodes
- By episode 90 the RL agent achieves positive rewards showing it has learned
- The rule-based agent is consistent but can get stuck navigating around obstacles
- The RL agent learns a more efficient path on familiar maps

---

## GitHub

[https://github.com/sadhvikoli/space-rover-agent](https://github.com/sadhvikoli/space-rover-agent)