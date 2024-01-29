import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SecurityEnvironment(gym.Env):
    ACTIONS = 10
    INFO = {'info': 'aaaa'}

    def __init__(self):
        super().__init__()
        self.MAX_STEPS = 50
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTIONS)
        self._reward = 0
        self._steps = 0
        self._episodes = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._episodes += 1
        obs, info = self.observation_space.sample(), self.INFO
        return obs, info

    def step(self, action):
        self._steps += 1
        return self.observation_space.sample(), 1, False, False, self.INFO

    def render(self):
        # return np.ones(self.observation_shape) * 1
        return None

    def close(self):
        return None
