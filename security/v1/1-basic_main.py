'''
Use a Custom Security environment as generic Gymnasium env for Reinforcement Learning.

This way, we can perform same techniques as with other Gym envs that we know to have good results.
'''

import gymnasium as gym
from security_environment import SecurityEnvironment
from time import sleep
from random import choice

# Gym env (in the future we will load it as other envs)
env = SecurityEnvironment()
actions = [0, 1, 2]  # available actions right now
observation, info = env.reset()
while (True):
    # for r in range(100):
    # sleep(1)
    # Try different actions and see result
    action = choice(actions)
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'Status: {observation[0]}')
    print(f'observation: {observation}')
    print(f'Reward: {reward}')
    if terminated or truncated:
        print(info)
        break
        env.reset()
env.close()
