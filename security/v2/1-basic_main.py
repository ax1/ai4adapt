
import gymnasium as gym
from security_environment import SecurityEnvironment

env = SecurityEnvironment()
observation, info = env.reset()
while (True):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
