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
import logging
import re
import dummy_environment as env

BASE_URL = 'http://localhost:8080/$TARGET/environment'
LOGGER_ENABLED = False
IDLE_ACTIONS_RATIO = 0.3    # Idle actions added to to the real ones to give more chances to learn do nothing when appropriate
MACHINES = 3                # Number of machines to train TODO implement in the future instead of having a separate file for dorothy


class REWARD(Enum):
    WIN = 100           # Some defenses have blocked all the attacks, end with FULL success.
    SURVIVE = 0         # After a while, if the system is still UP, end with success (is resilient)
    DIE = -100          # The attack destroys successfully the system
    USE_DEFENSE = -10   # The less weapons spent in defense, the better
    TIME = 0            # While still alive (even if damaged) the resilience is rewarded
    HEALTH = 0          # DO NOT USE, it give bad training results # When system is still healthy an extra reward is given
    SAVE_BULLETS = 10   # Reward when the agent should not move (do-nothing action) because nothing to defend yet
    # BLANK = 10


class SecurityEnvironment(gym.Env):

    def __init__(self, name='SecurityEnvironment', simulate=True, atomic=True):
        '''
        Start the environment. Note that the real env must be running.
        Params:
          name: [optional] the identifier of the model. Examples: 'MyEnv' or 'MyEnv_20K_3p_custom'
          simulate: [optional] use the external environment or use internal simulator for tests
          atomic: [optional] True, reward based on current step. False, sum previous rewards on each step.
          (note: atomic finds faster the right action, while cumulative discards faster ineffective actions)
        '''
        super().__init__()
        if (LOGGER_ENABLED):
            self._init_logger(name)
        current_target = name[:3]
        print(current_target)
        self._simulate = simulate
        self._atomic = atomic

        self._URL = BASE_URL.replace('$TARGET', current_target)
        print2()
        print2('----------------------------------------------------------------------------------------')
        print2('                       INIT Security Environment')
        print2('----------------------------------------------------------------------------------------')
        print2(f'RL agent: {name}') if name else None
        obj = env.info() if self._simulate else requests.get(self._URL).json()
        rewards_desc = [f'{el.name}: {el.value}' for el in REWARD]
        self.ACTIONS = obj['actions']
        self.OBSERVATIONS = obj['observations']
        actions_short = list(map(lambda el: f"{str(el['pos'])}-{el['name']} on {el['target']}", self.ACTIONS))
        print2(f'Atomic: {atomic}')
        print2(f'Rewards (strategy): {rewards_desc}')
        print2(f'Observations (targets): {self.OBSERVATIONS}')
        print2(f'Actions (defenses): {actions_short}')
        print2('----------------------------------------------------------------------------------------')
        self.OBSERVATION_NORMAL = 0
        self.OBSERVATION_COMPROMISED = 1
        self.OBSERVATION_DAMAGED = 2
        self.OBSERVATION_RESOLVED = 3
        # TODO @@@@@@@ Now, attack is much faster, we can reduce steps to consider invalid ending (currently 1 defense == 2 attacks in time)
        self.MAX_STEPS = 40
        self.action_space = spaces.Discrete(
            int(len(self.ACTIONS) * (1 + IDLE_ACTIONS_RATIO)))  # increase do nothing usage
        # Observations are [A,B,C] each with four states(0,1,2,3), where 0|3 are good and 1|2 are bad
        self.observation_space = spaces.Box(low=0, high=3, shape=(len(self.OBSERVATIONS),), dtype=np.uint8)
        self._reward = 0
        self._steps = 0
        self._episodes = 0
        self._last_observation = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._episodes += 1
        print2()
        print2(f'RESET Security Environment (Episode {self._episodes})')
        print2('Format: Observation before, A: Action performed, Observation after, R: Episode reward so far')
        print2('(Waiting 4 or 5 minutes for infrastructure to start...)')
        self._reward, self._steps = 0, 0
        res = env.reset() if self._simulate else requests.delete(self._URL).json()
        obs, info = res.values()
        print2(f'{info}. Initial observation: {obs}')
        observation = np.array(obs)
        self._last_observation = np.array(obs)
        self._result(-1, observation, self._reward, False, False, '')
        return observation, self._normalize_info(info)

    def step(self, action):
        # Initialize data
        self._reward = 0 if self._atomic else None
        terminated = False
        truncated = False
        if action >= len(self.ACTIONS):
            action = 0
        info = ''
        self._steps += 1
        obs = env.step(action) if self._simulate else requests.post(f'{self._URL}?action={action}').json()
        observation = np.array(obs)
        action_expensive = self.ACTIONS[action].get('name') != self.ACTIONS[0].get('name')
        was_damaged = self._is_damaged(self._last_observation)
        is_damaged = self._is_damaged(observation)

        # REWARD by time/step keeping alive the system
        self._update_reward(REWARD.TIME)

        # REWARD/PENALTY for action consumed
        #    Pay for action consumed
        self._update_reward(REWARD.USE_DEFENSE) if action_expensive else None
        #    Pay for inneficient action (can be disabled because efficient is success )
        self._update_reward(REWARD.USE_DEFENSE) if action_expensive and is_damaged else None
        #    Gain a tip on do nothing when system is ok (TODO: note that this uses a "previous" observation...)
        self._update_reward(REWARD.SAVE_BULLETS) if not action_expensive and not was_damaged else None
        #    But also pay for "loafing" instead of executing another defense, even a wrong one
        self._update_reward(REWARD.USE_DEFENSE) if not action_expensive and was_damaged else None

        # REWARD based on observation, if not many damages, give a tip
        damages = np.count_nonzero(observation)
        self._update_reward(REWARD.HEALTH, len(observation) - damages)

        # Check TRUNCATE based on SUCCESS damage control [not all VMs were attacked]
        if self._is_success(observation):
            info = f'{REWARD.WIN} (SUCCESS): The attack was stopped before damaging all the machines.'
            self._update_reward(REWARD.WIN)
            truncated = True
            return self._result(action, observation, self._reward, terminated, truncated, info)

        # Check TRUNCATE max steps (in this case is SUCCESS because the system is resilient to the attack)
        if self._steps >= self.MAX_STEPS:
            info = f'{REWARD.SURVIVE} (SUCCESS): The episode reached MAX_STEPS and the system is still ALIVE.'
            self._update_reward(REWARD.SURVIVE)
            truncated = True
            return self._result(action, observation, self._reward, terminated, truncated, info)

        # Check TERMINATE episode. In this case is FAILURE we terminate when last target is in unsafe state
        # if np.all(observation == self.OBSERVATION_DAMAGED):
        if observation[len(self.OBSERVATIONS) - 1] != self.OBSERVATION_NORMAL:
            info = f'{REWARD.DIE} (FAILURE): The last target is not in normal state. The global system is not secure.'
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

    def execute(self, action):
        '''
        Handy function to call env step() in inference mode so we do not consume RL steps
        '''
        obs = env.step(action) if self._simulate else requests.post(f'{self._URL}?action={action}').json()
        return obs

    def _is_damaged(self, obs):
        '''
        Check if system is considered as "damaged", or in bad state
        '''
        for item in obs:
            if item != 0 and item != 3:
                return True
        return False

    def _is_success(self, obs):
        '''
        If a target is 3 and the following ones are 0, the attack cannot develop anymore->SUCCESS
        BE CAREFUL when modifying this function because it is CRITICAL for training
        Note: Wizard cannot be considered 3=SUCCESS because already compromised
        '''
        if obs[1] == 3 and obs[2] == 0:
            return True
        elif obs[0] == 3 and obs[1] == 0 and obs[2] == 0:
            return True
        return False

    def action_desc(self, action):
        name = self.ACTIONS[action]['name']
        target = f"on {self.ACTIONS[action]['target']}" if "Do nothing" not in name else ''
        return 'Reset' if action == -1 else f"{name} {target}"

    def _result(self, action, observation, reward, terminated, truncated, info):
        print2(f'{str(self._steps).ljust(2)}:', self._last_observation, f'A{str(action).ljust(2)}', observation,
               f'R{str(reward).ljust(4)}', self.action_desc(action), info)
        self._last_observation = observation.copy()
        return observation, reward, terminated, truncated, self._normalize_info(info)

    def _update_reward(self, reward_type, times=1):
        self._reward = self._reward + (reward_type.value) * times

    def _init_logger(self, desc):
        filename = f"{desc}_{re.sub(r'(-|:)*','',str(datetime.now().isoformat(timespec='seconds')))}.log"
        FORMAT = '%(asctime)s %(message)s'
        logging.basicConfig(filename=filename, filemode='w', format=FORMAT,
                            level=logging.DEBUG, datefmt='%Y-%m-%dT%H:%M:%S')


def print2(*args):
    print(f"{datetime.now().isoformat(timespec='seconds')} ", *args) if args else print()
    # logging.info(*args) if (LOGGER_ENABLED) else None


'''
filename = f"desc_{re.sub(r'(-|:)*','',str(datetime.now().isoformat(timespec='seconds')))}.log"
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(filename=filename, filemode='a', format=FORMAT)
logger = logging.getLogger('tcpserver')
logger.info('aaa')
'''
