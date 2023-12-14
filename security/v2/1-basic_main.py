
import gymnasium as gym
from security_environment import SecurityEnvironment
from time import sleep


def print_line(type, observation, reward, terminated, truncated, info):
    return


COUNTER = 1
env = SecurityEnvironment()
observation, info = env.reset()
print(f'\nRESET (EPISODE {COUNTER})', observation, info)
while (True):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(action, observation, reward, terminated, truncated, info)
    if terminated or truncated:
        COUNTER += 1
        observation, info = env.reset()
        print(f'\nRESET (EPISODE {COUNTER})', observation, info)

env.close()
