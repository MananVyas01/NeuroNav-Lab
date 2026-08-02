# Neural Rocket Evolution

An interactive AI laboratory where neural-network-controlled rockets evolve to navigate toward a user-placed target through evolutionary optimization.

## Overview

Watch intelligence emerge as a population of 150 rockets learns to reach a target through evolutionary neural-network training. Starting from random behavior, the rockets progressively learn thrust control, rotation, and obstacle avoidance to navigate toward the goal.

## Features

- **Pure Neural Network**: NumPy implementation from scratch (no PyTorch/TensorFlow)
- **Evolutionary Training**: Tournament selection, crossover, mutation, elitism
- **Interactive Target**: Click anywhere to set/move the target
- **Real-time Visualization**: Neural network display, learning graphs, rocket trails
- **Multiple Obstacle Modes**: OFF, SIMPLE, MEDIUM, HARD, RANDOM, MAZE
- **Obstacle Sensors**: 5 ray sensors for environmental awareness
- **Speed Rewards**: Faster completion = higher fitness
- **Curriculum Mode**: Auto-increase difficulty based on success rate
- **Generalization Mode**: Randomly moving targets for general navigation learning
- **Save/Load**: Persist and restore trained networks
- **Adjustable Speed**: 1x to 25x simulation speeds

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| LEFT CLICK | Set target position |
| SPACE | Pause/Resume |
| R | Reset training |
| S | Save best network |
| L | Load saved network |
| +/- | Speed up/down |
| O | Cycle obstacle mode (OFF→SIMPLE→MEDIUM→HARD→RANDOM→MAZE) |
| N | Generate new RANDOM obstacle layout |
| T | Random target |
| G | Toggle generalization mode |
| C | Toggle curriculum mode |
| H | Toggle help |

## Neural Network Architecture

```
Input (14 neurons):
  - Normalized X, Y position (2)
  - Normalized velocity X, Y (2)
  - Normalized target X, Y (2)
  - Distance to target (1)
  - Angle to target (1)
  - Rocket orientation (1)
  - 5 obstacle sensor readings (5)

Hidden Layer 1: 20 neurons (ReLU)
Hidden Layer 2: 16 neurons (ReLU)

Output (2 neurons):
  - Turn (tanh → -1 to +1)
  - Thrust (tanh → 0 to 1)
```

## Obstacle Sensors

Each rocket has 5 forward-facing ray sensors:
- Far left (-0.8 rad)
- Front-left (-0.4 rad)
- Front (0.0 rad)
- Front-right (+0.4 rad)
- Far right (+0.8 rad)

Each sensor returns normalized distance (0.0 = very close, 1.0 = no obstacle).

## Obstacle Modes

| Mode | Description |
|------|-------------|
| OFF | No obstacles, pure navigation training |
| SIMPLE | 1-2 basic walls with large openings |
| MEDIUM | 3-5 obstacles requiring precise steering |
| HARD | Multiple walls, narrow passages, tight navigation |
| RANDOM | Randomized obstacle layouts |
| MAZE | Corridor-style maze with multiple turns |

## Evolution Algorithm

1. **Initialize**: Random weights for population of 150 networks
2. **Evaluate**: Run simulation for 600 frames per generation
3. **Score**: Calculate fitness based on rewards and completion speed
4. **Select**: Tournament selection (size 5)
5. **Crossover**: Uniform blending of parent weights (30% rate)
6. **Mutate**: Gaussian noise on weights (8% rate, 0.15 strength)
7. **Elitism**: Keep top 5% unchanged
8. **Repeat**

## Reward System

| Event | Reward |
|-------|--------|
| Getting closer | +2.0 |
| Moving away | -2.0 |
| Facing target | +0.5 |
| Reaching target | +1000 |
| Speed bonus | Up to +1500 for fast completion |
| Boundary collision | -100 |
| Obstacle crash | -300 |

### Speed Bonus

When a rocket reaches the target, it receives a speed bonus:
```python
speed_bonus = (GENERATION_LENGTH - steps_taken) / GENERATION_LENGTH * SPEED_REWARD
```

Faster completion = higher bonus. This encourages evolution to optimize for speed after learning basic navigation.

## Fitness Ranking

Rockets are naturally ranked:
1. Reached target extremely quickly (highest fitness)
2. Reached target quickly
3. Reached target slowly
4. Almost reached target (meaningful progress)
5. Made some progress
6. Made little progress
7. Crashed / failed badly (lowest fitness)

## Curriculum Mode

When enabled, training automatically increases difficulty:
- OFF → SIMPLE → MEDIUM → HARD → RANDOM → MAZE

Difficulty advances when success rate exceeds 75% for 5 consecutive generations.

## Project Structure

```
neural_rocket/
├── main.py              # Pygame UI and game loop
├── simulation.py        # Coordinates all components
├── rocket.py            # Rocket physics, sensors, state
├── neural_network.py    # Pure NumPy neural network
├── evolution.py         # Genetic algorithm
├── environment.py       # Target, obstacles, sensors
├── config.py            # All tunable parameters
├── models/              # Saved networks
└── README.md
```

## Expected Training Progress

| Generation | Behavior |
|------------|----------|
| 1-5 | Random movement, chaotic |
| 5-15 | Slight improvement, some drift |
| 15-30 | Basic thrust/turn control |
| 30-50 | Weak navigation patterns |
| 50-100 | Strong target seeking |
| 100+ | Efficient pathfinding |
| With obstacles | Learns sensor-based avoidance |

## Configuration

Edit `config.py` to tune:

- Population size and evolution parameters
- Physics constants
- Reward values
- Neural network architecture
- Sensor configuration
- Obstacle layouts
- Curriculum settings

## License

MIT
