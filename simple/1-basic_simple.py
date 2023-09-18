import gymnasium as gym
from SimpleEnv import SimpleEnv
from time import sleep
from random import choice

env = SimpleEnv()
actions = [-1, 0, 1]
observation, info = env.reset()
while (True):
    action = choice(actions)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'observation: {observation}')
    print(f'Reward: {reward}')
    if terminated or truncated:
        print(info)
        # break
        env.reset()
env.close()
