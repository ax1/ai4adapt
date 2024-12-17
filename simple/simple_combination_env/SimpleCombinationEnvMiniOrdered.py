import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleEnv(gym.Env):
    '''
    This is a shorter solution (2 switches). Use only for convergence tests.
    Then use the ordered version with 3 switches for comparison 

    Find a subset of switches to reach the solution. Ordered.
    Real usage can be find nodes to a path, etc, get the right combination, etc.

    Example: -v-v-----v--  -> OK
    '''
    ACTIONS = 10
    SOLUTION = [3, 7]
    MAX_STEPS = 10

    REWARD_ACTION = 2
    PENALTY_STEP = -1
    REWARD_SOLUTION = 10
    PENALTY_TIMEOUT = -10

    def __init__(self):
        super().__init__()
        # Observation is the current success status
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.ACTIONS)
        self._reward = 0
        self._steps = 0
        self._detected = None

    def _get_info(self, msg):
        return {'info': msg}

    def reset(self, seed=None, options=None):
        print('Reset')
        super().reset(seed=seed, options=options)
        self._reward = 0
        self._steps = 0
        initial = self.observation_space.sample()
        initial.fill(0)
        self._observation = initial
        info = self._get_info('System restarting')
        return self._observation, info

    def step(self, action):
        self._reward = 0
        terminated = False
        truncated = False
        self._steps += 1
        self._reward += self.PENALTY_STEP
        info = self._get_info(f'Step {self._steps}')
        observation = self._observation[0]
        if action == self.SOLUTION[observation]:
            info = self._get_info('Valid action found')
            observation += 1
            self._reward += self.REWARD_ACTION
        if self._steps >= self.MAX_STEPS:
            info = self._get_info('Truncate: Max steps reached')
            self._reward += self.PENALTY_TIMEOUT
            truncated = True
        elif observation == len(self.SOLUTION):
            info = self._get_info('Terminate: SUCCESS')
            self._reward += self.REWARD_SOLUTION
            terminated = True
        self._observation[0] = observation

        print(action, self._observation, self._reward, terminated, truncated, info)
        return self._observation, self._reward, terminated, truncated, info

    def render(self, mode):
        return None

    def close(self):
        return None
