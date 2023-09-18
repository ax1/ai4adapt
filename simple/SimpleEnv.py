import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traceback
from gymnasium.envs.registration import register


class SimpleEnv(gym.Env):
    """
    Simple environment for testing purposes.
    Scenario starts from a random int number, the target is a given fixed number.
    Actioms are do nothing increment and decrement
    """
    @staticmethod
    def register(id):
        entry_point = 'SimpleEnv:SimpleEnv'
        try:
            register(id=id, entry_point=entry_point,
                     max_episode_steps=SimpleEnv.MAX_STEPS)
        except:
            print(f'''ENVIRONMENT CLASS: Check if you imported the class properly (eg: from security_environment import *).
                   The expected path for this registration is \"{entry_point}\" ''')
            traceback.print_exc()

    MAX = 10
    TARGET = 3
    MAX_STEPS = 30

    def __init__(self):
        # max number [0,MAX]
        # self.observation_space = spaces.Discrete(self.MAX)
        self.observation_space = spaces.Box(
            low=0, high=self.MAX, shape=(1,), dtype=np.int32)
        # 0=-1 1=0 2=+1
        # LOOKOUT: SB3 learn has a bug, always [0,n] even i action is defined as start=-1
        self.action_space = spaces.Discrete(3)
        self._reward = 0
        self._steps = 0
        # self._observation = np.random.randint(self.MAX)

    def _get_info(self, msg):
        return {'info': msg}

    def reset(self, seed=None, options=None):
        self._reward = 0
        self._steps = 0
        # self._observation = np.array([np.random.randint(self.MAX)])
        self._observation = self.observation_space.sample()
        info = self._get_info(f'System restarting. TARGET is {self.TARGET}')
        return self._observation, info

    def step(self, action):
        terminated = False
        truncated = False
        self._steps += 1
        val = self._observation[0]+action-1
        self._observation[0] = val
        obs = self.observation_space.sample()
        self._reward += (-1)
        c = self._observation[0]
        t = self.TARGET
        info = self._get_info(f'Step count {self._steps}')

        if self._steps > self.MAX_STEPS:
            info = self._get_info('Truncate: Max steps reached')
            self._reward += (-100)
            truncated = True
        elif val < 0 or val >= self.MAX:
            info = self._get_info('Truncate: Out of bounds')
            self._reward += (-100)
            truncated = True
        elif c == t:
            info = self._get_info('Terminate: SUCCESS')
            self._reward += (+1000)
            terminated = True

        return self._observation, self._reward, terminated, truncated, info

    def render(self):
        # return np.ones(self.observation_shape) * 1
        return None

    def close(self):
        return None
