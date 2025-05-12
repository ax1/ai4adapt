'''
Simple environment for (mainly) DQN tests

UPDATE 08-04-2025. NO, the error was in the environment, I had an error in the reward so stay in 0 0 was rewarding positive.

Note that terminate=true or setting truncate=true instead has similar effects in training
and it shouldn't (one is for terminal state the other is for truncating before end), so we will need to check the same env in the future
against a different env framework and a different dqn framework to verify where is the potential problem
'''

import gymnasium as gym
import numpy as np


class SimpleCustomEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(5)
        self.state = 0

    def reset(self, seed=None, options=None):
        self.state = 0
        self.counter = 0
        return self.state, {}

    def step(self, action):
        last = self.state
        self.counter = self.counter + 1
        terminated = False
        truncated = False
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(4, self.state + 1)

        # Give reward only if advance to ANOTHER STATE (MARKOV 3 states, the backwards and the self states are penalties)
        reward = 1 if self.state > last else -1
        # reward = -1 if self.state < last else 1 # This was before and HAS AN ERROR because A0S0 gives reward

        if self.state == 4:
            reward = 10
            # This is the expected from docs, terminate to mark the leaf nodes
            terminated = True
            # but in SB3 there is some hidden error because if good reward strategy, truncate converges better
            # terminated = False
            # truncated = True
        elif self.counter >= 10:
            truncated = True

        return self.state, reward, terminated, truncated, {}

    def render(self, mode="console"):
        print(f"State: {self.state}")

    def close(self):
        pass
