import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traceback


class SimpleEnv(gym.Env):
    '''
    Find a subset of switches to reach the solution.
    Real usage can be find nodes to a path, etc
    Example: -v-v-----v--  -> OK
    '''
    ACTIONS = 10
    SOLUTION = [3, 7, 1]
    MAX_STEPS = 100

    def __init__(self):
        super().__init__()
        # Observation is the current success status
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.ACTIONS)
        self._reward = 0
        self._steps = 0
        self._solution = []

    def _get_info(self, msg):
        return {'info': msg}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reward = 0
        self._steps = 0
        zero = self.observation_space.sample()
        zero.fill(0)
        self._observation = zero
        self._solution = self.SOLUTION.copy()
        info = self._get_info(f'System restarting')
        return self._observation, info

    def step(self, action):
        terminated = False
        truncated = False
        self._steps += 1
        self._reward += (-1)
        if action in self._solution:
            self._solution.remove(action)
            self._observation[0] = self._observation[0] + 1/len(self.SOLUTION)
            self._reward += 1
        info = self._get_info(f'Step count {self._steps}')
        if self._steps >= self.MAX_STEPS:
            info = self._get_info('Truncate: Max steps reached')
            self._reward += (-10)
            truncated = True
        elif len(self._solution) == 0:
            info = self._get_info('Terminate: SUCCESS')
            self._reward += (+10)
            terminated = True
        return self._observation, self._reward, terminated, truncated, info

    def render(self, mode):
        return None

    def close(self):
        return None
