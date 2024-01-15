'''
See security_environment V1 as basis.

This module is kept with minimum code since lots of changes from 
time to time, so no need to leave reminder lines everywhere because it is
harder later for maintenance. For reference info go to security_environment.py file.

Note that all HTTP operations are sync, this is not an error, this is because the 
learning algorithms we're using for now are synchronous.
'''

import traceback
import gymnasium as gym
from gymnasium import spaces
from random import random
import subprocess
from enum import Enum
import numpy as np
import requests


URL = 'http://localhost:8080/environment'


class REWARD(Enum):
    TIMEOUT = 50        # After a while, if the system is still UP, end with success
    HEALED = 100        # Some defenses have blocked all the attack, end with full success
    FAILURE = -100      # The attack destroys successfully the system
    TRUNCATE = FAILURE  # When the system is fully DOWN, end episode with penalty
    DEFENSE = -1        # The less actions, the better
    TIME = 1            # While still alive (even if damaged) the resilience is rewarded
    HEALTH = 1          # When system is still healthy an extra reward is given

class SecurityEnvironment(gym.Env):

    def __init__(self):
        super().__init__()
        obj = requests.get(URL).json()
        self.ACTIONS = obj['actions']
        self.OBSERVATIONS = obj['observations']
        self.OBSERVATION_RESOLVED = 3
        self.OBSERVATION_DAMAGED = 2
        self.MAX_STEPS = 100
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # Observations are [A,B,C] each with four states(0,1,2,3), where 0|3 are good and 1|2 are bad
        self.observation_space = spaces.Box(low=0, high=3, shape=(len(self.OBSERVATIONS),), dtype=np.uint8)
        self._reward = 0
        self._steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reward, self._steps = 0, 0
        res = requests.delete(URL).json()
        obs, info = res.values()
        return np.array(obs), info

    def step(self, action):
        # Get all required data
        terminated = False
        truncated = False
        self._steps += 1
        info = f'Executing step {self._steps}'
        obs = requests.post(f'{URL}?action={action}').json()
        observation = np.array(obs)
        # observation = self.observation_space.sample()

        # REWARD by time/step keeping alive the system
        self._update_reward(REWARD.TIME)

        # REWARD/PENALTY for action consumed
        if action != 0:
            self._update_reward(REWARD.DEFENSE)

        # REWARD based on observation, if not many damages, give a tip
        damages = np.count_nonzero(observation)
        self._update_reward(REWARD.HEALTH, len(observation) - damages)

        # Check TRUNCATE episode (in this case is SUCCESS because the system is resilient to the attack)
        if self._steps > self.MAX_STEPS:
            info = 'TIMEOUT (SUCCESS): The episode reached MAX_STEPS and the system is still ALIVE.'
            self._update_reward(REWARD.TIMEOUT)
            truncated = True
            return observation, self._reward, terminated, truncated, info

        # Check TRUNCATE based on SUCCESS damage control [ALL VMs in state protected=val3]
        if np.all(observation == self.OBSERVATION_RESOLVED):
            info = 'HEALED (SUCCESS): The episode was resolved with all items protected after the attack.'
            self._update_reward(REWARD.HEALED)
            truncated = True
            return observation, self._reward, terminated, truncated, info

        # Check TERMINATE episode. In this case is FAILURE we terminate when all the system is down
        if np.all(observation == self.OBSERVATION_DAMAGED):
            info = 'END (FAILURE): All the observations are system damages'
            self._update_reward(REWARD.FAILURE)
            terminated = True
            return observation, self._reward, terminated, truncated, info

        # If no truncated or terminated, send the new observation for the agent to proceed
        return observation, self._reward, terminated, truncated, info

    def render(self):
        return np.ones(self.observation_shape) * 1
        # return None

    def close(self):
        return None

    def _update_reward(self, reward_type, times=1):
        self._reward = self._reward + (reward_type.value) * times
