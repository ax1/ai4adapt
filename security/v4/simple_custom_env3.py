import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleCustomEnv(gym.Env):
    """
    A simple environment where the agent moves in a 1D space from state 0 to 9.
    The goal is to reach state 9 as fast as possible.
    """

    def __init__(self):
        super(SimpleCustomEnv, self).__init__()

        # State space: 10 discrete states (0 to 9)
        self.observation_space = spaces.Discrete(10)

        # Action space: 2 actions (0 = decrease, 1 = increase)
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.state = 0
        self.steps = 0
        self.max_steps = 20  # Prevent infinite loops

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """
        self.state = 0
        self.steps = 0
        return self.state, {}

    def step(self, action):
        """
        Executes an action and returns the next state, reward, and termination flag.
        """
        self.steps += 1

        # Apply action: move left or right
        if action == 1 and self.state < 9:
            self.state += 1
        elif action == 0 and self.state > 0:
            self.state -= 1

        # Reward system
        reward = -1  # Small penalty per step to encourage efficiency
        done = False

        # Success condition
        if self.state == 9:
            reward = 10  # Large reward for reaching the goal
            done = True

        # Terminate if max steps reached
        if self.steps >= self.max_steps:
            done = True

        return self.state, reward, done, False, {}

    def render(self):
        print(f"State: {self.state}")

    def close(self):
        pass
