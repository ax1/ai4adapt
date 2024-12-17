import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleEnv(gym.Env):
    '''
    Find a subset of switches to reach the solution. No order of switch, just enable the required ones.
    Real usage can be find nodes to a path, etc, get the right combination, etc.

    Example: -v-v-----v--  -> OK

    Note: Be careful, this is an example of how a simple problem can be very hard to train with RL.

    Note: this is HARD for PPO and maybe others because prob to find abc is 1000 and we
    are training with 1000-2000 stochastic steps so big problem with iterations.
    Another problem is that for the RL there are lots of good combinations so if setting
    predict to deterministic=True will try forever only one eg:7 because 7 was good in all
    observed states, not only in the second one.
    '''
    ACTIONS = 10
    SOLUTION = [3, 7, 1]
    MAX_STEPS = 10
    REWARD_ACTION = 2
    PENALTY_STEP = -1
    REWARD_SOLUTION = 2
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
        self._detected = set()
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
        info = self._get_info(f'Step count {self._steps}')
        if action in self.SOLUTION and action not in self._detected:
            info = self._get_info('Reward: valid action found')
            self._detected.add(action)
            self._observation[0] = self._observation[0] + 1
            self._reward += self.REWARD_ACTION
        if self._steps >= self.MAX_STEPS:
            info = self._get_info('Truncate: Max steps reached')
            self._reward += self.PENALTY_TIMEOUT
            truncated = True
        elif len(self._detected) == len(self.SOLUTION):
            info = self._get_info('Terminate: SUCCESS')
            self._reward += self.REWARD_SOLUTION
            terminated = True
        print(action, self._observation, self._reward, terminated, truncated, info)
        return self._observation, self._reward, terminated, truncated, info

    def render(self, mode):
        return None

    def close(self):
        return None
