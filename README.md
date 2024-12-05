# LiteSocNavGym

**LiteSocNavGym** is a lightweight Gymnasium environment designed for social navigation tasks involving robots, humans, and tables. The environment simulates scenarios where a robot navigates toward a goal while avoiding collisions with humans and static obstacles (tables).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running an Example](#running-an-example)
- [Environment Details](#environment-details)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Rewards](#rewards)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Dynamic Agents:** Both robots and humans move toward their respective goals.
- **Collision Avoidance:** Implements collision detection and avoidance mechanisms.
- **Customizable Parameters:** Adjust the number of humans, tables, rewards, and more.
- **Efficient Rendering:** Visualize the environment using OpenCV with minimal resource usage.
## Environment Details

### Observation Space

- **Robot's Orientation:**
  - `[cos(theta), sin(theta)]` if `robot_view` is `True`.
  - `[x, y, cos(theta), sin(theta)]` if `robot_view` is `False`.

- **Goal's Position:**
  - Relative to the robot if `robot_view` is `True`.
  - Absolute position if `robot_view` is `False`.

- **Humans' Positions and Orientations:**
  - Relative to the robot if `robot_view` is `True`.
  - Absolute positions and orientations if `robot_view` is `False`.

- **Tables' Positions and Sizes:**
  - Relative to the robot if `robot_view` is `True`.
  - Absolute positions and sizes if `robot_view` is `False`.

- **Masks:**
  - Indicate active humans and tables.

### Action Space

The action space is continuous:

- **[advance, rotation]**
  - Advance: Controls the forward movement of the robot.
  - Rotation: Controls the rotational movement of the robot.

For the discrete version (`DiscreteLiteSocNavGym`), the action space is discrete with predefined action mappings.

### Rewards

- **Reach Reward:** +1.0 for reaching the goal.
- **Out of Map Reward:** -0.5 for moving out of bounds.
- **Max Ticks Reward:** -0.3 for exceeding the maximum number of steps.
- **Alive Reward:** -0.003 as a living penalty to encourage faster goal achievement.
- **Collision Reward:** -0.5 for collisions with humans or tables.
- **Discomfort Penalty:** Additional penalties for being too close to humans.




## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` package manager




### Clone the Repository

```bash
git clone https://github.com/goodluckoguzie/LiteSocNavGym.git
cd LiteSocNavGym

