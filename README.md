# Fuzzy-Guided-Reinforcement-Learning-AGV
# Fuzzy-Guided Reinforcement Learning for Sim2Real Transfer in Multi-Robot System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen.svg)](http://wiki.ros.org/noetic)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3%2B-orange.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A novel architecture that deeply integrates **Fuzzy Logic** with **Reinforcement Learning** to achieve safe, efficient Sim2Real transfer for multi-robot navigation systems.

**aiRobots Laboratory**, Department of Electrical Engineering, National Cheng Kung University, Tainan, Taiwan

---

## Overview

This project proposes a **Fuzzy-Guided Reinforcement Learning (Fuzzy-RL)** framework that combines expert knowledge (via fuzzy inference) with autonomous learning (via RL) to address critical challenges in deploying robots from simulation to real-world environments.

### Key Contributions

- **Deep integration** of fuzzy logic into RL architecture with 44 expert-designed fuzzy rules
- **Hierarchical Fuzzy-RL architecture** (Strategic → Tactical → Execution layers)
- **Multi-layer safety assurance** mechanism for safe real-world deployment
- **Multi-robot transfer learning** framework for cross-platform knowledge sharing
- **Three-phase Sim2Real transfer** strategy (Simulation → Progressive Realization → Real-World Fine-tuning)

### Why Fuzzy-Guided RL?

| Challenge (Traditional RL) | Solution (Fuzzy-Guided RL) |
|---|---|
| Low exploration efficiency | Safety-constrained exploration with expert guidance |
| Poor safety during training | Fuzzy-based action filtering & multi-layer safety |
| Slow learning convergence | Fuzzy-enhanced reward shaping |
| Poor interpretability | Transparent fuzzy rule-based decision making |
| Difficult Sim2Real transfer | Three-phase transfer with fuzzy rule adaptation |

---

## Architecture

### Hierarchical Fuzzy-RL

```
┌─────────────────────────────────────────────────┐
│  High Level: Fuzzy Strategic Controller          │
│  (Expert Knowledge + Safety/Exploration/Behavior)│
├─────────────────────────────────────────────────┤
│  Middle Level: RL Tactical Agent                 │
│  (Policy Network + Value Network + Replay Buffer)│
├─────────────────────────────────────────────────┤
│  Low Level: Adaptive Motor Controller            │
│  (Motion Control + Path Planning + Safety Monitor)│
└─────────────────────────────────────────────────┘
```

### Fuzzy Inference System — Four Guidance Dimensions

| Dimension | Input Variables | Output | Mechanism |
|---|---|---|---|
| Safety Assessment | Obstacle distance, Velocity | `Safety_Level` | Action filtering, exploration restrictions |
| Exploration Guidance | Goal distance, Environment complexity | `Exploration_Encouragement` | Adjust exploration probability |
| Confidence Regulation | Goal angle, Task difficulty | `Action_Confidence` | Action magnitude adjustment |
| Speed Control | Environment state, Task urgency | `Speed_Modifier` | Direct speed regulation |

### Fuzzy-Guided Actor-Critic Network

The RL agent fuses a **10-dim state input** with a **4-dim fuzzy guidance signal** through an attention mechanism, producing both policy actions and state value estimates.

```
State Input (10-dim) ──┐
                       ├──→ Feature Fusion ──→ Policy Network ──→ Action Output
Fuzzy Signal (4-dim) ──┤                  ──→ Value Network  ──→ State Value
                       └──→ Attention Mechanism ──→ Weight Adjustment
```

---

## Project Structure

```
Fuzzy-Guided-RL-AGV/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── default.yaml              # Default training hyperparameters
│   ├── fuzzy_rules.yaml          # Fuzzy rule definitions (44 rules)
│   └── robot_configs/
│       ├── ammr.yaml             # AMMR robot configuration
│       └── circular_amr.yaml    # Circular AMR configuration
├── agent/
│   ├── __init__.py
│   ├── ppo_agent.py              # PPO-based RL agent
│   ├── actor_critic.py           # Fuzzy-Guided Actor-Critic network
│   ├── replay_buffer.py          # Experience replay buffer
│   └── safety_filter.py          # Action safety filtering module
├── fuzzy/
│   ├── __init__.py
│   ├── inference_system.py       # Fuzzy Inference System (FIS)
│   ├── membership_functions.py   # Triangular membership function definitions
│   ├── rule_base.py              # 44 fuzzy rules (Safety/Exploration/Confidence/Speed/Attention/Compound)
│   └── defuzzification.py        # Weighted average defuzzification
├── envs/
│   ├── __init__.py
│   ├── base_env.py               # Base environment interface
│   └── mujoco/
│       ├── __init__.py
│       ├── navigation_env.py     # Navigation task environment
│       ├── multi_robot_env.py    # Multi-robot coordination environment
│       └── assets/
│           ├── ammr.xml          # AMMR MuJoCo model
│           ├── circular_amr.xml  # Circular AMR MuJoCo model
│           ├── chimei_demo.xml   # Chimei 2F Demo Room scene
│           └── chimei_corridor.xml # Chimei 2F Corridor scene
├── transfer/
│   ├── __init__.py
│   ├── sim2real.py               # Three-phase Sim2Real transfer pipeline
│   ├── robot_abstraction.py      # Robot characteristic vector abstraction
│   ├── rule_adaptation.py        # Fuzzy rule adaptation for new robots
│   └── domain_randomization.py   # Noise injection for robustness
├── safety/
│   ├── __init__.py
│   ├── safety_monitor.py         # Multi-layer safety architecture
│   └── collision_predictor.py    # Trajectory-based collision prediction
├── data/
│   ├── logs/                     # Training logs & TensorBoard
│   ├── checkpoints/              # Model checkpoints
│   └── results/                  # Evaluation results
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation & metrics
│   ├── sim2real_transfer.py      # Sim2Real transfer pipeline
│   └── visualize.py              # Result visualization
└── tests/
    ├── test_fuzzy.py             # Fuzzy inference system tests
    ├── test_agent.py             # RL agent tests
    └── test_safety.py            # Safety module tests
```

---

## Reward Design

### Base Reward

```
R_base = R_goal + R_collision + R_efficiency
```

| Component | Formula | Coefficient |
|---|---|---|
| Goal Distance Penalty | `R_goal = -α × d_goal` | α = 0.1 |
| Collision Penalty | `R_collision = -β × I_collision` | β = 10.0 |
| Action Cost | `R_efficiency = -γ × |a_t|` | γ = 0.01 |

### Fuzzy Enhanced Reward

```
R_fuzzy = R_safety + R_exploration + R_confidence
```

| Component | Formula | Coefficient |
|---|---|---|
| Safety Reward | `λ₁ × safety_level(f_t) × I_near_obstacle` | λ₁ = 0.05 |
| Exploration Reward | `λ₂ × exploration_encouragement(f_t) × novelty(s_t)` | λ₂ = 0.02 |
| Confidence Reward | `λ₃ × action_confidence(f_t) × success_rate` | λ₃ = 0.03 |

### Total Reward

```
R_total = R_base + R_fuzzy + R_adaptive
R_adaptive = η × sign(performance_trend) × |fuzzy_contribution|    (η = 0.01)
```

---

## Fuzzy Rule Base

The system contains **44 fuzzy rules** across 6 categories:

| Category | # Rules | Function | Weight Range |
|---|---|---|---|
| Safety Rules | 8 | Collision avoidance, risk assessment | 0.6 – 1.0 |
| Exploration Strategy Rules | 8 | Learning efficiency optimization | 0.6 – 0.9 |
| Confidence Rules | 8 | Decision certainty regulation | 0.5 – 0.9 |
| Speed Control Rules | 9 | Motion efficiency & safety balance | 0.6 – 1.0 |
| Attention Rules | 6 | Dynamic computational resource allocation | 0.5 – 0.9 |
| Compound Rules | 5 | Multi-constraint situation handling | 0.7 – 0.9 |

**Example Rules:**
- `IF obstacle = VERY_NEAR THEN safety_level = CRITICAL` (weight: 1.0)
- `IF goal_distance = FAR AND obstacle_distance = FAR THEN exploration_encouragement = HIGH` (weight: 0.8)
- `IF velocity = FAST AND obstacle_distance = NEAR THEN action_confidence = LOW` (weight: 0.9)

---

## Multi-Layer Safety Architecture

| Level | Mechanism | Trigger | Response |
|---|---|---|---|
| Level 1 | Fuzzy safety assessment | `safety_level < 0.3` | Limit action space |
| Level 2 | Physical constraint check | Exceed motion limits | Action clipping |
| Level 3 | Collision prediction | Predicted trajectory danger | Emergency stop |
| Level 4 | Human intervention | Continuous anomalies | System pause |

---

## Sim2Real Transfer

### Three-Phase Strategy

1. **Phase 1 — Simulation Training**: Train base policy in perfect simulation → Apply fuzzy rules
2. **Phase 2 — Progressive Realization**: Inject noise → Robustness testing → Rule adjustment
3. **Phase 3 — Real-World Fine-tuning**: Deploy to real robot → Fine-tune → Online update

### Multi-Robot Transfer Pipeline

```
Source Robot (trained) → Characteristic Mapping → Rule Adaptation → Policy Fine-tuning → Target Robot (deployed)
```

Robot Description Vector = `[Morphological Features, Dynamic Parameters, Sensor Configuration, Task Capabilities]`

---

## Experiment Setup

### Hardware
- **AMMR** (Autonomous Mobile Manipulator Robot)
- **Circular AMR** (Autonomous Mobile Robot)

### Simulator
- **MuJoCo + ROS**

### Scenarios
- Chimei 2F Demo Room
- Chimei 2F Corridor

### Baselines
| Method | Type | Comparison Focus |
|---|---|---|
| Pure PPO | Pure RL | Overall performance |
| Domain Randomization | Sim2Real | Transfer effectiveness |
| Progressive Networks | Transfer | Transfer speed |
| Pure Fuzzy Control | Traditional | Adaptability |
| Constrained RL | Safe RL | Safety |

### Target Metrics

| Metric | Target |
|---|---|
| Convergence speed | < 1000 episodes |
| Collision rate | < 2% |
| Transfer sample efficiency | < 100 samples |
| Task success rate | > 90% |
| Path efficiency (actual/shortest) | < 1.2 |

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# MuJoCo
# Follow: https://mujoco.org/download

# ROS Noetic (for real robot deployment)
# Follow: http://wiki.ros.org/noetic/Installation
```

### Training

```bash
# Train with default config (Fuzzy-Guided PPO)
python scripts/train.py --config config/default.yaml

# Train baseline (Pure PPO, no fuzzy guidance)
python scripts/train.py --config config/default.yaml --no-fuzzy

# Train with specific robot
python scripts/train.py --robot ammr
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint data/checkpoints/best_model.pt

# Run ablation study
python scripts/evaluate.py --ablation
```

### Sim2Real Transfer

```bash
# Execute three-phase transfer
python scripts/sim2real_transfer.py --source circular_amr --target ammr
```

---

## State & Action Space

### State Input (10-dim)

| Variable | Description | Range |
|---|---|---|
| `position_x` | Current x-coordinate | — |
| `position_y` | Current y-coordinate | — |
| `orientation` | Current heading | [0, 2π] |
| `linear_velocity` | Linear speed | [0, 1] m/s |
| `angular_velocity` | Angular speed | — |
| `goal_x` | Goal x-coordinate | — |
| `goal_y` | Goal y-coordinate | — |
| `min_lidar_distance` | Nearest obstacle distance | [0, 10] m |
| `avg_lidar_distance` | Average LiDAR range | [0, 10] m |
| `goal_angle` | Relative angle to goal | [-π, π] rad |

### Fuzzy Guidance Output (4-dim)

| Signal | Range | Description |
|---|---|---|
| `safety_level` | [0, 1] | Current safety assessment |
| `exploration_encouragement` | [0, 1] | Exploration encouragement degree |
| `action_confidence` | [0, 1] | Decision confidence level |
| `speed_modifier` | [0, 1] | Speed adjustment factor |

---

## Future Work

- Integration with **Large Language Models (LLMs)** for high-level task understanding and plan generation
- Extension to more robot types beyond AMR/AMMR
- Online fuzzy rule learning and auto-tuning

---

## References

1. Shuzheng Qu, Mohammed Abouheaf, Wail Gueaieb, and Davide Spinello, "An Adaptive Fuzzy Reinforcement Learning Cooperative Approach for the Autonomous Control of Flock Systems," 2023, [arXiv:2303.09946](https://arxiv.org/abs/2303.09946)

2. Yuwei Fu, Haichao Zhang, Di Wu, Wei Xu, Benoit Boulet, "FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning," 2024, [arXiv:2406.00645](https://arxiv.org/abs/2406.00645)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{chen2025fuzzyrl,
  title={Fuzzy-Guided Reinforcement Learning for Sim2Real Transfer in Multi-Robot System},
  author={Meng-Xue Chen},
  year={2025},
  institution={National Cheng Kung University}
}
```
