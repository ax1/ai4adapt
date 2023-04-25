import gymnasium as gym
from SecurityEnvironment import SecurityEnvironment
from time import sleep

env = SecurityEnvironment()

for r in range(100):
    sleep(1)
    res = env.step(1)
    print(res)
