import gymnasium as gym
from gymnasium import spaces
from random import random
import subprocess
from enum import Enum


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

    For observations, in this first stage, the status of the system is the only parameter(0-ok, 1-distressed)

    For actions, the space is a discrete set of numbers, and not an action object,
        the object is dealt internally only. For external users, actions are only a list of numbers.

    For rewards:
        - Negative reward:
          - each defense executed. Amortization/cost of installed solutions,resources like electricity space, cloud resources , etc).
          - AND each step() (time passed) that the system is vulnerable. In other envs this is the time/gas/cost consumed.
          - AND timeout: After several steps (enough time for attacker to blow the system or steal data) the system is considered broken. Very negative big reward.
        - Positive reward:
          - if system status is OK (attack was resolved, solution was found). Moderate big reward.
    '''

    MAX_STEPS = 20  # More than 10 unsuccesful defenses for a single attack is not realistic as defense

    """
     ██████  ██ ███    ███ ███    ██  █████  ███████ ██ ██    ██ ███    ███     ███    ███ ███████ ████████ ██   ██  ██████  ██████  ███████
    ██       ██ ████  ████ ████   ██ ██   ██ ██      ██ ██    ██ ████  ████     ████  ████ ██         ██    ██   ██ ██    ██ ██   ██ ██
    ██   ███ ██ ██ ████ ██ ██ ██  ██ ███████ ███████ ██ ██    ██ ██ ████ ██     ██ ████ ██ █████      ██    ███████ ██    ██ ██   ██ ███████
    ██    ██ ██ ██  ██  ██ ██  ██ ██ ██   ██      ██ ██ ██    ██ ██  ██  ██     ██  ██  ██ ██         ██    ██   ██ ██    ██ ██   ██      ██
     ██████  ██ ██      ██ ██   ████ ██   ██ ███████ ██  ██████  ██      ██     ██      ██ ███████    ██    ██   ██  ██████  ██████  ███████
    """

    def __init__(self):
        # Note: indicators is optional, but set for usage in the future (eg: net, cpu, ram...)
        # @Eider: check wich indicators based on attack, for now status if enough
        self.observation_space = spaces.Dict({
            'status': spaces.Discrete(1),
            'indicators': spaces.Discrete(4)
        })
        self._actions = self._load_actions()
        # Action number for external users (1,2,3...)
        self.action_space = spaces.Discrete(len(self._actions))
        self._initialize_damaged_environment()
        self._reward = 0
        self._steps = 0

    def _get_obs(self):
        status = self._get_system_observation()['status']
        return {'status': status, 'metrics': [random(), random(), random(), random()]}

    def _get_info(self, msg):
        return {f'info: {msg}'}

    def reset(self, seed=None, options=None):
        self._reward = 0
        self._steps = 0
        self._initialize_damaged_environment()
        # This initialize numpy.random (as other envs do, not required by us)
        super().reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info('System restarting')
        return observation, info

    def step(self, action):
        terminated = False
        truncated = False

        # General step
        self._steps += 1
        info = self._get_info(f'Executing step #{self._steps}')
        self._update_reward(REWARD.TIME)

        # Check abort (out of time, out of cost, etc..) and reaturn early
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

        # After the action, observe the enviroment to see the new state
        observation = self._get_obs()

        # If objective is achieved, notify to finish as SUCCESS
        if observation['status'] == STATUS.GREEN.value:
            info = self._get_info('SUCCESS: status OK')
            self._update_reward(REWARD.SUCCESS)
            terminated = True

        # Return feedback to the agent
        return observation, self._reward, terminated, truncated, info

    def render(self):
        return None

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
        Either run on the real environment infrastructure or request I/O externally
        '''
        dummy_status = 0 if random() > 0.9 else 1  # simulate distressed most of the time
        return {
            'status': dummy_status,
            'network_data': 'event,pattern',
            'operational_data': 'pattern,threshold...',
            'device_data': 'CPU,RAM,DISK...',
            # @Eider: timestamp will be useful later to fine-tune time cost instead of step=time
            'timestamp': 1682515800959
        }

    def _update_reward(self, reward_type):
        self._reward = self._reward + reward_type.value
