import gymnasium as gym
from gymnasium import spaces
from random import random


class SecurityEnvironment(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Dict({})
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        return {"agent": random(), "target": random()}

    def _get_info(self):
        return {"distance": random()}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)  # TODO
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = random()
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = random()
        # An episode is done iff the agent has reached the target
        terminated = True if random() > 0.95 else False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        return None

    def close(self):
        return None


env = SecurityEnvironment()
res = env.step(1)
print(res)
