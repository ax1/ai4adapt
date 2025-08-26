# AI4ADAPT

Reinforcement Learning (RL) for security defences optimization.

The aim of this project is to use popular RL tools, like [Gymnasium](https://gymnasium.farama.org/) for environment simulations and [Stable Baselines3 (SB3)](https://stable-baselines3.readthedocs.io), for Deep Learning training and use real environments instead of simulators or digital twins to perform the training. The agents created with this method are more reliable to handle security issues when deployed into production. Highlights:
- A Gym-like custom environment for security activities is provided, so people with knowledge of RL frameworks can use it with little extra knowledge. 
- The environment is prepared to connect to real digital systems where a cybersecurity expert and a devops engineer may be required depending on the complexity of the security problem.

The project has been tested to successfully defend against Advanced Persistent Threats (APTs), where an attacker silently takes control of a vulnerable machine in the system and then attempts to escalate access to other machines with higher privileges. Once the attacker reaches the final target (typically a server), they could steal information, impersonate members of the organization, or encrypt the server and demand ransom.

## Installation

> DEV status: check inside example files depending on the wanted lib for custom installation instructions.

```sh

# Clone repo
git clone https://github.com/ax1/ai4adapt
cd ai4adapt

# Optional, create a virtual env
python -m venv .venv
source ./.venv/bin/activate

#---------- a) complete installation -----
# aiohttp only required to use security_env_proxy. For Python 3.10+ asiyncio is built-in 
# gymnasium[box2d] is optional (install only box2d to use the gym sample envs), if so, swig is also required
pip install swig gymnasium[box2d] aiohttp stable-baselines3[extra]

#---------- b) minimal installation ------
# Use this instead when planning to run in a headless server, or just the security environment (in server there is no GUI, so installing swig+box2d will fail)
pip install gymnasium aiohttp stable-baselines3[extra]

#---------- c) contributors and maintainers ------
# Use the requirements.txt file to use the same version of libraries
pip install -r requirements.txt

```

## Usage

> See README.md in each folder

- security folder: SecurityEnvironment.py created as a gym env. Examples on how to use it. Also a dummy environment proxy is available to run before the real environment is available.
- examples/Lunar folder: minimal examples on how to use gym environments (no RL,RL, DRL).
- examples/Bar2D folder: another set of examples, mainly for assessing the DRL libs on a simple env. 
- examples/simple folder: collection of custom crafted environments resolving different security problems in a generic way. Pre-tuning the hyperparameters for the security environment is usually faster using these environments. Additionally, they provide insights on how the agent is learning and the minimum length of iterations that will be required later (running things targeting real machines will be always slower than iterating with these environments).

```sh
from security_environment import SecurityEnvironment

# Create environment
env = SecurityEnvironment()

# [OPTIONAL] register and use as registered env
ENV_NAME = 'security_env'
SecurityEnvironment.register(ENV_NAME)
env = gym.make(ENV_NAME)

```

> Note: the "default" project setup runs the environment by using a dummy simulator. In case the environment proxy (ai4adapt_env) is already available, just toggle the flag `SIMULATE` to `false`.

## Reward strategy

The behaviour of the trained agent will be heavily influenced by the set of rewards to be configured in the `security_environment.py` file. By default, choosing the right defense at a given state provides good reward, but learning to wait idle (do-nothing) when the attack has not been detected yet is important. Also, there is a high reward when the agent stops any further evolution of the attack, thus promoting the agent to find first how to stop the attack and then tune the less costly way to stop it.

## Training report files

The `security_environment.py` is already prepared to generate a documentation with the training setup and the result of the execution. Run any agent and pass the standard output to a file.
That information is provided in structured format (for parsers) but also in human readable format to understand how the agent was evolving from random execution to refined defensive strategies. 

## Creation of the real environment proxy

We have developed another project, that is currently in managed private repository [AI4ADAPT_env](https://github.com/ax1/ai4adapt_env) with proprietary license, but creating a proxy is straightforward by implementing a REST service with the following endpoints:
- GET ${url} : equivalent to Gymnasium init(). Return initialization information (the list of actions, the observation space)
- DELETE ${url}: equivalent to Gymnasium reset(). Delete any old state in the server, or reboot machines, and return initial info message to the agent.
- POST ${url} : equivalent to Gymnasium step(). Receive the action id (defense), to execute in the real system and return the observations after the execution.

## Resources

AI4ADAPT has been created during the execution of the UE project [AI4CYBER](https://ai4cyber.eu/). There are many useful resources and documentation available:
- [AI4ADAPT research paper](https://doi.org/10.1016/j.eswa.2025.129168)
- [AI4CYBER website and documentation](https://ai4cyber.eu/?page_id=62)
- [AI4CYBER at Cordis](https://cordis.europa.eu/project/id/101070450)

## Contact

[AI4CYBER contact page](https://ai4cyber.eu/?page_id=219/#contact)


## Acknowledgements

|<img src="https://ai4cyber.eu/wp-content/uploads/2022/12/Funded-Europe.png" alt="Funded by Europe" width="200"/>|This work has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070450 (AI4CYBER).|
|---|---|

