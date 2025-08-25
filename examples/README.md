# General purpose examples for AI4ADAPT

This folder contains a set of files to serve different purposes:
- Known Gym/Gymnasium environments: Useful to test classical or DRL algorithms on general purpose environments. This allows later to tune the security environment based on insights provided by these environments. Mainly used for assessing Deep Reinforcement Learning behaviour (for instance DQN finds good solutions but PPO is by far the first finding the optimal solution).
- Generic environments for multi-combination problems (folder `simple`): Small environments with few actions/ observations, they are perfect for pre-tuning hyperparameters for the security environment, since the type of challenge for the agent is similar and we can perform plenty of executions instead of dealing with the real environment or with harder environments as the ones in Gymnasium.
