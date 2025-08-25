import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traceback
from gymnasium.envs.registration import register


class SimpleEnv(gym.GoalEnv):
    """
    Env adapted for goal type algorithms. Read "HER" at https://openai.com/research/ingredients-for-robotics-research

     Go to THIS value (given a slider)  
    |--------█----------------------|

    Simple environment for testing purposes. It behaves as a slider finding a fixed position
    Scenario starts from a random int number, the target is a given fixed number in the slider.
    Actions are {decrement, do nothing, increment}
    """
    @staticmethod
    def register(id):
        entry_point = 'SimpleSliderEnv:SimpleEnv'
        try:
            register(id=id, entry_point=entry_point,
                     max_episode_steps=SimpleEnv.MAX_STEPS)
        except:
            print(f'''ENVIRONMENT CLASS: Check if you imported the class properly (eg: from security_environment import *).
                   The expected path for this registration is \"{entry_point}\" ''')
            traceback.print_exc()

    LOWER_BOUND = 0
    UPPER_BOUND = 9  # 10 steps
    TARGET = 3
    MAX_STEPS = 100

    def __init__(self):
        # max number [0,9]
        self.observation_space = spaces.Box(
            low=self.LOWER_BOUND, high=self.UPPER_BOUND, shape=(1,), dtype=np.int32)
        # 0=-1 1=0 2=+1
        # LOOKOUT: As defined in documentation SB3  Discrete is not Integer,
        #  but NATURAL numbers {0,N} So start=-1 is not possible even if not runtime error.
        self.action_space = spaces.Discrete(3)

        self._reward = 0
        self._steps = 0

    def _get_info(self, msg):
        return {'info': msg}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reward = 0
        self._steps = 0
        self._observation = self.observation_space.sample()
        # self._observation = np.array([0]).astype(np.int32)
        info = self._get_info(f'System restarting. TARGET is {self.TARGET}')
        return self._observation, info

    def step(self, action):
        slide = action-1  # -1,0,1
        # #
        # obs = self._observation[0]
        # if obs == 0 and slide == -1:
        #     print('WILL TERMINATE LEFT')
        # if obs >= 9 and slide == +1:
        #     print('WILL TERMINATE RIGHT')
        # #

        terminated = False
        truncated = False
        self._steps += 1
        old_observation = self._observation[0]
        val = self._observation[0]+slide
        self._observation[0] = val
        self._reward += 10-abs(val-self.TARGET)
        c = self._observation[0]
        t = self.TARGET
        info = self._get_info(f'Step count {self._steps}')

        if self._steps > self.MAX_STEPS:
            info = self._get_info('Truncate: Max steps reached')
            self._reward += (-100)
            truncated = True
        elif val < self.LOWER_BOUND or val > self.UPPER_BOUND:
            info = self._get_info('Truncate: Out of bounds')
            self._reward += (-100)
            truncated = True
            self._observation[0] = old_observation
        elif c == t:
            info = self._get_info('Terminate: SUCCESS')
            self._reward += (+100)
            terminated = True

        return self._observation, self._reward, terminated, truncated, info

    def render(self, mode):
        # return np.ones(self.observation_shape) * 1
        return None

    def close(self):
        return None
