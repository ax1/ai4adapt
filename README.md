# AI4ADAPT

Reinforcement Learning (RL) for security defences optimization.

## Installation

```sh
pip install gymnasium[box2d]
```

## Usage

> See README.md in each folder

- Security folder: SecurityEnvironment.py created as a gym env. Examples on how to use it.
- Lunar folder: minimal examples on how to use gym environments (no RL).
- Bar2D folder: another set of examples, mainly for assessing the DRL libs on a simple env. 

```sh
from SecurityEnvironment import *

# Create environment
env = SecurityEnvironment()

# [OPTIONAL] register and use as registered env
ENV_NAME = 'security_env'
SecurityEnvironment.register(ENV_NAME)
env = gym.make(ENV_NAME)

```

Read also https://xusophia.github.io/DataSciFinalProj/

## Approach (12/09/2023)

After assessing pros and cons, we have decided to split the project into 2 objectives:
- The RL training for security: keep it simple, do not try to re-invent the wheel. There are many literature on RL for security. So we separate the training from the environment as much as possible. This is achieved by the SecurityEnvironment module.
- The environment: Create a Real environment with real initial attack scenario and with real sensing and REAL defenses. Pass the sensing to the training part as gymnasium env and simple/clear observations and rewards. Even if the RL does not create the optimal strategy, having a real system gives more confidence in the training than using a simulator

Additionally: to limit the scope we will train only reactive defenses (not the preventive ones). In the future we can put also preventive defenses to the trained ones so the system will be more robust and not only able to solve the attacks on the fly.