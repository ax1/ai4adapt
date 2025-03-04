'''
DQN example for small iterations environments
Note that either exploration_final_eps=1 to force as much random exploration or train_freq=(5, 'episode')
or similar for exploration are required when max_steps is short
(in this case a Q-learning is enough to solve these env problems but we're exploring Deep Learning as generalization)
'''

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from simple_custom_env import SimpleCustomEnv

custom_env = SimpleCustomEnv()

# this helps:  exploration_final_eps=1 but it is better to force train every n episodes train_freq=(5, 'episode')
model = DQN("MlpPolicy", custom_env, verbose=1, exploration_final_eps=1)
# model = DQN("MlpPolicy", custom_env, verbose=1)
model.learn(total_timesteps=512, progress_bar=False)

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
# print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()
steps = 0
while (steps < 40):
    steps += 1
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    print('action', action, 'obs', obs, 'reward', rewards, 'dones', dones)
