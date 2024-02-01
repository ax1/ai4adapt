'''
(See security_environment V1 as basis).

This module is kept with minimum code since lots of changes from
time to time, so no need to leave reminder lines everywhere because it is
harder later for maintenance. For reference info go to security_environment.py file.

Note that all HTTP operations are sync, this is not an error, this is because the
learning algorithms we're using for now are synchronous.
'''

import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import numpy as np
import requests
from datetime import datetime

URL = 'http://localhost:8080/environment'


class REWARD(Enum):
    # @@@@@ TODO NOW WE WILL HAVE ATOMIC 3 EG [0,3,0], SO GIFT ALSO AS PARTIAL WIN
    WIN = 100           # Some defenses have blocked all the attacks, end with FULL success.
    SURVIVE = 0         # After a while, if the system is still UP, end with success (is resilient)
    DIE = -100          # The attack destroys successfully the system
    USE_DEFENSE = -1    # The less weapons spent in defense, the better
    TIME = 1            # While still alive (even if damaged) the resilience is rewarded
    HEALTH = 1          # When system is still healthy an extra reward is given


class SecurityEnvironment(gym.Env):

    def __init__(self, description=None):
        super().__init__()
        print2()
        print2('----------------------------------------------------------------------------------------')
        print2('                       INIT Security Environment')
        print2('----------------------------------------------------------------------------------------')
        print2(f'RL agent: {description}') if description else None
        obj = requests.get(URL).json()
        rewards_desc = [f'{el.name}: {el.value}' for el in REWARD]
        self.ACTIONS = obj['actions']
        self.OBSERVATIONS = obj['observations']
        actions_short = list(map(lambda el: f"{str(el['pos'])}-{el['name']} on {el['target']}", self.ACTIONS))
        print2(f'Rewards (strategy): {rewards_desc}')
        print2(f'Observations (targets): {self.OBSERVATIONS}')
        print2(f'Actions (defenses): {actions_short}')
        print2('----------------------------------------------------------------------------------------')
        self.OBSERVATION_NORMAL = 0
        self.OBSERVATION_COMPROMISED = 1
        self.OBSERVATION_DAMAGED = 2
        self.OBSERVATION_RESOLVED = 3
        self.MAX_STEPS = 10    # Our current attack is 48 steps
        self.action_space = spaces.Discrete(12)
        # Observations are [A,B,C] each with four states(0,1,2,3), where 0|3 are good and 1|2 are bad
        self.observation_space = spaces.Box(low=0, high=3, shape=(len(self.OBSERVATIONS),), dtype=np.uint8)
        self._reward = 0
        self._steps = 0
        self._episodes = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._episodes += 1
        print2()
        print2(f'RESET Security Environment (Episode {self._episodes})')
        print2('A: Action, O: Observation, R: Reward')
        print2('(Waiting 4 or 5 minutes for infrastructure to start...)')
        self._reward, self._steps = 0, 0
        res = requests.delete(URL).json()
        obs, info = res.values()
        print2(f'{info}. Initial observation: {obs}')
        return np.array(obs), self._normalize_info(info)

    def step(self, action):
        action = 0
        # print2(f'Executing action {action}-{self.action_desc(action)} ...')
        # Get all required data
        terminated = False
        truncated = False
        info = ''
        self._steps += 1
        obs = requests.post(f'{URL}?action={action}').json()
        observation = np.array(obs)
        # observation = self.observation_space.sample()

        # REWARD by time/step keeping alive the system
        self._update_reward(REWARD.TIME)

        # REWARD/PENALTY for action consumed
        if action != 0:
            self._update_reward(REWARD.USE_DEFENSE)

        # REWARD based on observation, if not many damages, give a tip
        damages = np.count_nonzero(observation)
        self._update_reward(REWARD.HEALTH, len(observation) - damages)

        # Check TRUNCATE episode (in this case is SUCCESS because the system is resilient to the attack)
        if self._steps >= self.MAX_STEPS:
            info = f'{REWARD.SURVIVE} (SUCCESS): The episode reached MAX_STEPS and the system is still ALIVE.'
            self._update_reward(REWARD.SURVIVE)
            truncated = True
            return self._result(action, observation, self._reward, terminated, truncated, info)

        # Check TRUNCATE based on SUCCESS damage control [not all VMs were attacked]
        if self._is_success(observation):
            info = f'{REWARD.WIN} (SUCCESS): The attack was stopped before damaging all the machines.'
            self._update_reward(REWARD.WIN)
            truncated = True
            return self._result(action, observation, self._reward, terminated, truncated, info)

        # Check TERMINATE episode. In this case is FAILURE we terminate when all the system is down
        if np.all(observation == self.OBSERVATION_DAMAGED):
            info = f'{REWARD.DIE} (FAILURE): All the observations are critical damages. The system cannot be recovered.'
            self._update_reward(REWARD.DIE)
            terminated = True
            return self._result(action, observation, self._reward, terminated, truncated, info)

        # If no truncated or terminated, send the new observation for the agent to proceed
        return self._result(action, observation, self._reward, terminated, truncated, info)

    def render(self):
        return np.ones(self.observation_shape) * 1
        # return None

    def close(self):
        return None

    def _normalize_info(self, info):
        return {'info': f'{info}'}

    def _is_success(self, obs):
        '''
        If a target is 3 and the following ones are 0, the attack cannot develop anymore->SUCCESS
        '''
        for r in range(len(obs)):
            if obs[r] == 3:
                remain = []
                for s in range(r + 1, len(obs)):
                    if obs[s] == 0 or obs[s] == 3:
                        remain.append(obs[s])
                if (len(remain) == len(obs) - r - 1):
                    return True
        return False

    def action_desc(self, action):
        return f"{self.ACTIONS[action]['name']} on {self.ACTIONS[action]['target']}"

    def _result(self, action, observation, reward, terminated, truncated, info):
        print2(f'{str(self._steps).ljust(2)}:', f'A{str(action).ljust(2)}', f'O{observation}',
               f'R{str(reward).ljust(3)}', self.action_desc(action), info)
        return observation, reward, terminated, truncated, self._normalize_info(info)

    def _update_reward(self, reward_type, times=1):
        self._reward = self._reward + (reward_type.value) * times


def print2(*args):
    print(f"{datetime.now().isoformat(timespec='seconds')} ", *args) if args else print()
