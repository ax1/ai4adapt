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
   still installation error and enough fast with tensorflow and cuda y itself so no need to install)
'''

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

NAME = "CartPole-v1"

# Parallel environments (n_envs)
vec_env = make_vec_env(NAME, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=2500)
model.save(NAME)

del model  # remove to demonstrate saving and loading

model = PPO.load(NAME)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    # TODO ARF: with newer envs, done is replaced by terminated truncated
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
