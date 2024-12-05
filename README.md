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

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` package manager

### Clone the Repository

```bash
git clone https://github.com/goodluckoguzie/LiteSocNavGym.git
cd LiteSocNavGym

