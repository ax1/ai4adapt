import gymnasium as gym
from gymnasium import spaces
from random import random
import subprocess
from enum import Enum


class STATUS(Enum):
    GREEN = 0
    RED = 0


class SecurityEnvironment(gym.Env):
    '''
    Create a Gym environment and follow their rules (when possible) for external interaction.
    The

    For observations, in this first stage, the status of the system is the ony parameter

    For actions, the space is a discrete set of numbers, and not an action object,
        the object is dealt internally only. For external users, actions are only a list of numbers.

    For rewards: 
        penalty for:
        - each defense executed
        - AND each step() (time passed) that the system is vulnerable. In othere envs this is the gas consumed.
        awards:
        - if system status is OK (attack was resolved)
    '''

    """
        ██████╗██╗   ██╗███╗   ███╗███╗   ██╗ █████╗ ███████╗██╗██╗   ██╗███╗   ███╗    ███╗   ███╗███████╗████████╗██╗  ██╗ ██████╗ ██████╗ ███████╗
        ██╔════╝╚██╗ ██╔╝████╗ ████║████╗  ██║██╔══██╗██╔════╝██║██║   ██║████╗ ████║    ████╗ ████║██╔════╝╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗██╔════╝
        ██║  ███╗╚████╔╝ ██╔████╔██║██╔██╗ ██║███████║███████╗██║██║   ██║██╔████╔██║    ██╔████╔██║█████╗     ██║   ███████║██║   ██║██║  ██║███████╗
        ██║   ██║ ╚██╔╝  ██║╚██╔╝██║██║╚██╗██║██╔══██║╚════██║██║██║   ██║██║╚██╔╝██║    ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║   ██║██║  ██║╚════██║
        ╚██████╔╝  ██║   ██║ ╚═╝ ██║██║ ╚████║██║  ██║███████║██║╚██████╔╝██║ ╚═╝ ██║    ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝███████║
        ╚═════╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝     ╚═╝    ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝                                                                                                                                           
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

    def _get_obs(self):
        status = self._get_system_status()['status']
        return {'status': status, 'indicators': [0, 0, 0, 0]}

    def _get_info(self):
        return {"info": None}

    def reset(self, seed=None, options=None):
        # This initialize numpy.random (as other envs do, not required by us)
        super().reset(seed=seed)

        self._initialize_damaged_environment()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self._execute_action(action)

        # ---OLD-----DELETE LATER---------------------
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = random()
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = random()
        # An episode is done if the agent has reached the target
        terminated = True if random() > 0.95 else False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        return None

    def close(self):
        return None

    """
    ███████╗███████╗ ██████╗██╗   ██╗██████╗ ██╗████████╗██╗   ██╗    ███╗   ███╗███████╗████████╗██╗  ██╗ ██████╗ ██████╗ ███████╗
    ██╔════╝██╔════╝██╔════╝██║   ██║██╔══██╗██║╚══██╔══╝╚██╗ ██╔╝    ████╗ ████║██╔════╝╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗██╔════╝
    ███████╗█████╗  ██║     ██║   ██║██████╔╝██║   ██║    ╚████╔╝     ██╔████╔██║█████╗     ██║   ███████║██║   ██║██║  ██║███████╗
    ╚════██║██╔══╝  ██║     ██║   ██║██╔══██╗██║   ██║     ╚██╔╝      ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║   ██║██║  ██║╚════██║
    ███████║███████╗╚██████╗╚██████╔╝██║  ██║██║   ██║      ██║       ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝███████║
    ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝       ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
    """

    def _load_actions(self):
        # @Eider: we must run a search response items or list them somehow.
        print('Loading actions: TODO, in the future, this list should be a call to the available actions (Responses)')
        actions = {}
        actions[0] = {'id': 0, 'name': 'NO ACTION',
                      'command': 'echo NO ACTION'}
        actions[1] = {'id': 10023, 'name': 'firewall',
                      'command': 'echo start firewall'}
        actions[2] = {'id': 40001, 'name': 'restart',
                      'command': 'echo restarting (reboot) the machine'}
        return actions

    def _execute_action(self, num):
        # Be careful on real envs or disable shell
        subprocess.run(self._actions[num]['command'], shell=True)

    def _initialize_damaged_environment(self):
        subprocess.run(
            'echo TODO put environment in distressed mode or restart distressed default environment', shell=True)

    def _get_system_status(self):
        '''
        Either run on the real environment infrastructure or request I/O externally
        '''
        dummy_status = 0 if random() > 0.90 else 1  # simulate distressed most of the time
        return {'status': dummy_status, 'network_data': 'event,pattern', 'operational_data': 'pattern,threshold...', 'device_data': 'CPU,RAM,DISK...'}
