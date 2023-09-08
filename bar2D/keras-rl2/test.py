'''
Delete this file when solved the gym in bar2d problem

https://discuss.tensorflow.org/t/typeerror-keras-symbolic-inputs-outputs-do-not-implement-len/15355
'''

import csv
import numpy as np
import gym
from gym import spaces
from typing import List
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class MyEnv(gym.Env):
    def __init__(self):
        # initialize your environment here
        self.profit = None
        self.params = None
        self.database = download_market_data()
        self.observation_space = gym.spaces.Box(low=np.array(
            [0.0, 1]), high=np.array([0.1, 25]), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.current_step = 0
        self.max_steps = 1000

    def step(self, action):
        # take an action (which corresponds to selecting some random parameters)
        # and return the next state, reward, and done flag
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.current_step < self.max_steps, "Max steps reached. Call reset to start again."

        self.current_step += 1

        params = self.observation_space.sample()
        flat = params.flatten()

        database = find_extrema(self.database, flat[0], flat[1])
        profit = calculate_profitability(database)

        reward = profit
        done = self.current_step == self.max_steps
        self.params = params
        self.profit = profit
        return params, reward, done, {}

    def reset(self, **kwargs):
        # reset the environment to its initial state
        self.current_step = 0
        return self.observation_space.sample()

    def render(self, mode='human'):
        print(f"params: {self.params}, profit: {self.profit}")


# Define the environment
env = MyEnv()

# Define the number of actions and the number of observations
nb_actions = env.action_space.n
nb_observations = env.observation_space.shape[0]

# Define the model architecture
model = tf.keras.models.Sequential()
model.add(InputLayer(input_shape=(nb_observations,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))

# Define the memory buffer
memory = SequentialMemory(limit=1000000, window_length=1)

# Define the policy for choosing actions
policy = EpsGreedyQPolicy(0.1)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Define the DQN agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True, enable_dueling_network=True,
               dueling_type="avg")

# Compile the DQN agent with optimizer
dqn.compile(optimizer=Adam(learning_rate=1e-3))

# Train the DQN agent
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
