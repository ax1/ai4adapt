'''
SUMMARY: PROBLEMS WITH COMPATIBILIY of libs (many tests tried but only solution is to downgrade tf to v1 and we do not want this)
CONCLUSION: read as example but do not develop this file

Basic gym bar example with neural networks for RL.

See the bar2D_basic.py for code comparison.

Use a TensorFlow RNN for training.

Installation requirements: gymnasium[box2d] (+ tensorflow) + keras-rl2 (note the '2' otherwise keras error due to keras-rl how to solve https://discuss.tensorflow.org/t/typeerror-keras-symbolic-inputs-outputs-do-not-implement-len/15355)
'''

import random
import gymnasium as gym
from keras import Sequential
import keras.layers as layers
import keras.optimizers as optimizers
import rl
import rl.agents as agents
import rl.policy as policy
import rl.memory as memory
import numpy as np


def create_model(env):    # TODO check flatten layer really required for or scenarios
    len_observation = env.observation_space.shape[0]
    len_actions = env.action_space.n
    model = Sequential()
    # TODO check flatten layer really required for "or" scenarios
    # model.add(layers.Flatten(input_shape=(1, len_observation)))
    model.add(layers.Flatten(input_shape=(1,  2)))
    # model.add(layers.Dense(32, activation='relu',input_shape=env.observation_space.shape))
    # still error but check for comparison https://discuss.tensorflow.org/t/typeerror-keras-symbolic-inputs-outputs-do-not-implement-len/15355
    # read also https://stackoverflow.com/questions/67000544/valueerror-error-when-checking-input-expected-dense-input-to-have-2-dimensions
    # model.add(layers.InputLayer(input_shape=(len_observation,)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    # model.add(layers.Flatten())
    model.add(layers.Dense(len_actions, activation='linear'))
    return model


def create_agent(env):
    model = create_model(env)
    agent = agents.DQNAgent(
        model=model,
        memory=memory.SequentialMemory(limit=10000, window_length=1),
        policy=policy.BoltzmannQPolicy(),
        nb_actions=env.action_space.n, nb_steps_warmup=10, target_model_update=0.01)
    agent.compile(optimizers.Adam(lr=0.01), metrics=["mae"])
    return agent


def main():
    env = gym.make('CartPole-v1', render_mode='human')
    agent = create_agent(env)
    agent.fit(env, nb_steps=10000, visualize=False, verbose=1)
    # results = agent.test(env, nb_episodes=10, visualize=True)
    # print(np.mean(results.history["episode_reward"]))
    env.close()


main()
