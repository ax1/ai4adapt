import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traceback
from gymnasium.envs.registration import register


class SimpleEnv(gym.Env):
    """
    Simple environment for testing purposes.
    Instead of the slider, this is a directional problem.
    Left to bound is success, right bound is fail
    Note that penalty for step has been disabled because when using
    PPO algorithm later, the learning steps are very high when the 
    step penalty is -10 (even if low compared to success).
    Tested with -1 also worked but better to disable. 
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
    LOWER_BOUND = 0
    UPPER_BOUND = 9
    MAX_STEPS = 30

    def __init__(self):
        super().__init__()
        # max number [0,9]
        self.observation_space = spaces.Box(
            low=0, high=self.UPPER_BOUND, shape=(1,), dtype=np.int32)
        # 0=-1 1=+1
        # LOOKOUT: As defined in documentation SB3  Discrete is not Integer,
        #  but NATURAL numbers {0,N} So start=-1 is not possible even if not runtime error.
        self.action_space = spaces.Discrete(2)
        self._reward = 0
        self._steps = 0

    def _get_info(self, msg):
        return {'info': msg}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reward = 0
        self._steps = 0
        self._observation = self.observation_space.sample()
        info = self._get_info(f'System restarting')
        return self._observation, info

    def step(self, action):
        observation = None
        terminated = False
        truncated = False
        self._steps += 1
        '''
        BE CAREFUL with moderate step penalty values.
        For instance -1 is fine with 1000 training steps
        However -10 is really bad for 1000 or 10000 training steps
        '''
        # self._reward += (-1)
        info = self._get_info(f'Step count {self._steps}')

        if self._steps >= self.MAX_STEPS:
            info = self._get_info('Truncate: Max steps reached')
            self._reward += (-100)
            truncated = True
        elif action == 1 and self._observation[0] >= self.UPPER_BOUND:
            info = self._get_info('Truncate: Out of bounds')
            self._reward += (-100)
            truncated = True
        elif action == 0 and self._observation[0] <= self.LOWER_BOUND:
            info = self._get_info('Terminate: SUCCESS')
            self._reward += (+1000)
            terminated = True
        self._observation[0] = self._observation[0] - \
            1 if action == 0 else self._observation[0]+1
        return self._observation, self._reward, terminated, truncated, info

    def render(self, mode):
        # return np.ones(self.observation_shape) * 1
        return None

    def close(self):
        return None
