'''
Example from https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

See also the available ppo policies in the above documentation link. basically:
- stable_baselines3.ppo.MlpPolicy (alias of stable_baselines3.common.policies.ActorCriticPolicy)
- stable_baselines3.ppo.CnnPolicy (alias of stable_baselines3.common.policies.ActorCriticCnnPolicy)
- stable_baselines3.ppo.MultiInputPolicy (alias of stable_baselines3.common.policies.MultiInputActorCriticPolicy)


Installation:
- [optional] install cuda
 -pip install stable_baselines3 
 -[optional pip install (tensorrt install eror, recommends to pip install --extra-index-url https://pypi.nvidia.com tensorrt-libs,
   still installation error and enough fast with tensoflow and cuda y itself so no need to install)
'''

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=2500)
model.save("ppo_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
