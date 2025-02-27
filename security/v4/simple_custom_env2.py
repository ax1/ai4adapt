import gymnasium as gym
import numpy as np


class SimpleCustomEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(5)
        self.state = 0

    def reset(self, seed=None, options=None):
        self.state = 0
        return self.state, {}

    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(4, self.state + 1)

        reward = -1
        if self.state == 4:
            reward = 10
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self, mode="console"):
        print(f"State: {self.state}")

    def close(self):
        pass
