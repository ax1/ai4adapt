import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleCustomEnv(gym.Env):
    def __init__(self):
        super(SimpleCustomEnv, self).__init__()
        # Define the action space: 2 actions (0 or 1)
        self.action_space = spaces.Discrete(2)
        # Define the observation space: a single scalar state
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Initialize the state
        self.state = None
        self.steps = 0
        self.max_steps = 10  # Limit the episode length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state to a random value between 0 and 1
        self.state = np.array([np.random.rand()], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        # Reward the agent for taking action 1, penalize for action 0
        reward = 1 if action == 1 else -1
        # Check if the episode is done
        done = self.steps >= self.max_steps
        # The next state is random (no specific dynamics in this simple example)
        self.state = np.array([np.random.rand()], dtype=np.float32)
        return self.state, reward, done, False, {}

    def render(self):
        pass  # No rendering for this simple environment

    def close(self):
        pass
