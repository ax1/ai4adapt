'''
Example using LUNAR + PPO + SB3

In this case , check other policies to compare 

Note: check also the lunar PPO to compare (same degree of skill)
- number of iterations
- total time to learn
- convergence speed to solution
- overfitting (by looking at glitches on some episodes)

Results:
- PPO 250K (same with 100K) iterations not perfect but at least the training spots that the optimal should be to land (DQN example in this folder did not achieve that)
'''


import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_NAME = "LunarLander-v2"
LEARN_ITERATIONS = 250000
MODEL_NAME = ENV_NAME+'_'+str(LEARN_ITERATIONS)

# Parallel environments

vec_env = make_vec_env(ENV_NAME, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=LEARN_ITERATIONS)
model.save(MODEL_NAME)

del model  # remove to demonstrate saving and loading

model = PPO.load(MODEL_NAME)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    # TODO ARF: with newer envs, done is replaced by terminated truncated
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
