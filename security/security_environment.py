import traceback
import gymnasium as gym
from gymnasium import spaces
from random import random
import subprocess
from enum import Enum
from gymnasium.envs.registration import register
import numpy as np


class STATUS(Enum):
    GREEN = 0
    RED = 0


class REWARD(Enum):
    # @Eider: Real values for security issues should be bigger than these ones (breach) depending on the attack
    TIME = -2
    DEFENSE = -1
    TIMEOUT = -10000
    SUCCESS = 100


class SecurityEnvironment(gym.Env):
    '''
    Create a Gym environment and follow their rules (when possible) for external interaction.

    Observation: array (Box1D) of numbers (See doc for match to real sensing).

    Action: array (Discrete) from 0-n (See doc for match to real defense name).

    For rewards:
        - Negative reward:
          - each defense executed. Amortization/cost of installed solutions,resources like electricity space, cloud resources , etc).
          - AND each step() (time passed) that the system is vulnerable. In other envs this is the time/gas/cost consumed.
          - AND timeout: After several steps (enough time for attacker to blow the system or steal data) the system is considered broken. Very negative big reward.
        - Positive reward:
          - if system status is OK (attack was resolved, solution was found). Moderate big reward.

    Usage:
     - as normal code: `env = new SecurityEnvironment()`
     - as registered env: `SecurityEnvironment.register('SecurityEnvironment')` and then `gym.make('SecurityEnvironment)`
    '''

    MAX_STEPS = 50  # More than 10-50 unsuccessful defenses for a single attack is not realistic as real defense
    """
     ██████  ██ ███    ███ ███    ██  █████  ███████ ██ ██    ██ ███    ███     ███    ███ ███████ ████████ ██   ██  ██████  ██████  ███████
    ██       ██ ████  ████ ████   ██ ██   ██ ██      ██ ██    ██ ████  ████     ████  ████ ██         ██    ██   ██ ██    ██ ██   ██ ██
    ██   ███ ██ ██ ████ ██ ██ ██  ██ ███████ ███████ ██ ██    ██ ██ ████ ██     ██ ████ ██ █████      ██    ███████ ██    ██ ██   ██ ███████
    ██    ██ ██ ██  ██  ██ ██  ██ ██ ██   ██      ██ ██ ██    ██ ██  ██  ██     ██  ██  ██ ██         ██    ██   ██ ██    ██ ██   ██      ██
     ██████  ██ ██      ██ ██   ████ ██   ██ ███████ ██  ██████  ██      ██     ██      ██ ███████    ██    ██   ██  ██████  ██████  ███████
    """
    @staticmethod
    def register(id):
        '''
        This method is OPTIONAL, since the env can be created as `env = SecurityEnvironment()`.
        Other RL libraries accept either a String or the ClassName as input so register is not really mandatory.
        The method registers *at runtime* the current env, so it can be invoked from gym.make().

        The try/except below is a handy guard to remind how to import for a successful registering
        (this is better than copying files to gym folder).
        '''
        entry_point = 'security_environment:SecurityEnvironment'
        try:
            register(id=id, entry_point=entry_point,
                     max_episode_steps=SecurityEnvironment.MAX_STEPS)
        except:
            print(f'''ENVIRONMENT CLASS: Check if you imported the class properly (eg: from security_environment import *).
                   The expected path for this registration is \"{entry_point}\" ''')
            traceback.print_exc()

    def __init__(self):
        super().__init__()
        # Note: indicators is optional, but set for usage in the future (eg: net, cpu, ram...)
        # @Eider: check wich indicators based on attack, for now status if enough
        self.observation_space = spaces.Discrete(5)
        self._actions = self._load_actions()
        # Action number for external users (1,2,3...)
        self.action_space = spaces.Discrete(len(self._actions))
        self._initialize_damaged_environment()
        self._reward = 0
        self._steps = 0

    def _get_obs(self):
        # status = self._get_system_observation()[0]
        return self._get_system_observation()
        # return {'status': status, 'metrics': [random(), random(), random(), random()]}

    def _get_info(self, msg):
        return {'info': msg}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reward = 0
        self._steps = 0
        self._initialize_damaged_environment()
        observation = self._get_system_observation()
        info = self._get_info('System restarting')
        return observation, info

    def step(self, action):
        terminated = False
        truncated = False

        # General step
        self._steps += 1
        info = self._get_info(f'Executing step #{self._steps}')
        self._update_reward(REWARD.TIME)

        # Check abort (out of time, out of cost, etc..) and return early
        if self._steps > self.MAX_STEPS:
            info = self._get_info(
                'TIMEOUT: The current set of defenses wer noth enough fast to solve the attack.')
            self._update_reward(REWARD.TIMEOUT)
            truncated = True
            observation = self._get_obs()
            return observation, self._reward, terminated, truncated, info

        # Check if action performed or no-action
        if action != None:
            self._update_reward(REWARD.DEFENSE)
            self._execute_action(action)

        # After the action, observe the environment to see the new state
        observation = self._get_obs()

        # If objective is achieved, notify to finish as SUCCESS
        status = observation[0]
        if status == STATUS.GREEN.value:
            info = self._get_info('SUCCESS: status OK')
            self._update_reward(REWARD.SUCCESS)
            terminated = True

        # Return feedback to the agent
        return observation, self._reward, terminated, truncated, info

    def render(self):
        return np.ones(self.observation_shape) * 1
        # return None

    def close(self):
        return None

    """
     ██████ ██    ██ ███████ ████████  ██████  ███    ███     ███    ███ ███████ ████████ ██   ██  ██████  ██████  ███████
    ██      ██    ██ ██         ██    ██    ██ ████  ████     ████  ████ ██         ██    ██   ██ ██    ██ ██   ██ ██
    ██      ██    ██ ███████    ██    ██    ██ ██ ████ ██     ██ ████ ██ █████      ██    ███████ ██    ██ ██   ██ ███████
    ██      ██    ██      ██    ██    ██    ██ ██  ██  ██     ██  ██  ██ ██         ██    ██   ██ ██    ██ ██   ██      ██
     ██████  ██████  ███████    ██     ██████  ██      ██     ██      ██ ███████    ██    ██   ██  ██████  ██████  ███████
    """

    def _load_actions(self):
        # @Eider: we must run a search response items or list them somehow.
        print('Loading actions: TODO, in the future, this list should be a call to the available actions (Responses)')
        actions = {}
        actions[0] = {'id': 0, 'name': 'NO ACTION',
                      'command': 'echo DEFENSE: NO ACTION'}
        actions[1] = {'id': 10023, 'name': 'firewall',
                      'command': 'echo DEFENSE: start firewall'}
        actions[2] = {'id': 40001, 'name': 'restart',
                      'command': 'echo DEFENSE: stopping the machine'}
        return actions

    def _execute_action(self, num):
        # Be careful on real envs or disable shell
        subprocess.run(self._actions[num]['command'], shell=True)

    def _initialize_damaged_environment(self):
        subprocess.run(
            'echo TODO put environment in distressed mode or restart distressed default environment', shell=True)
        # TODO this will involve deleting and generating a new VM with the environment
        # and also to start in attacked mode so initial status should be 1

    def _get_system_observation(self):
        '''
        Either run on the real environment infrastructure or request I/O externally.
        ARF 15/09/23: translate dict into an array (another option is to use MultiInputSpace policy instead of MLPPolicy later).
        This is NOT mandatory but useful later in the DRL algorithm (usually requires a value or array of values)
        Note that for the env itself, it is better to use the dict (so visually direct match to system parameters), but we want env to be gym_like
        because we want to demonstrate that we can learn the same way a standard gym env (lunar, etc) and then with the same code, the custom env.
        '''
        dummy_status = 0 if random() > 0.9 else 1  # simulate distressed most of the time
        observation = {
            # Note: these parameters are passed later to the agent as a Dictionary (as environment "metrics" )
            # @Eider check status how can be extracted from the system
            'status': dummy_status,
            'network_data': 'event,pattern',
            'operational_data': 'pattern,threshold...',
            'device_data': 'CPU,RAM,DISK...',
            # @Eider: timestamp will be useful later to fine-tune time cost instead of step=time
            'timestamp': 1682515800959
        }
        # TODO in the future the DICT is a Dict<SPACES>, not a Dict(any) See https://gymnasium.farama.org/api/spaces/composite/
        # return gym.spaces.Dict(observation)  # only for MultiInputPolicy
        # return gym.spaces.MultiDiscrete([dummy_status, 1, 2, 3, 4])
        return np.array([3]) @ @@TODO esto funciona mirar arriba como he definido el discrete(que significa 1 valor de shape y que como maximo puede valer de 0 a 4) ver discrete aqui https: // gymnasium.farama.org/api/spaces/fundamental / y tambien ver el ejemplo de luarn que usa un box en el obervation y un discrete en el action en lunar y en cartpole lo mismo a mod de ejemplo https: // github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py https: // github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        # return gym.spaces.Discrete(1, 1)
        # return observation.values()  # list(observation.values())

    def _update_reward(self, reward_type):
        self._reward = self._reward + reward_type.value
