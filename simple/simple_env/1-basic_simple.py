import gymnasium as gym
from SimpleEnv import SimpleEnv
from time import sleep
from random import choice

env = SimpleEnv()
observation, info = env.reset()
while (True):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'action {action} Reward: {reward} next_observation {observation}')
    if terminated or truncated:
        print(info)
        env.reset()
env.close()
