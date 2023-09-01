'''
Basic gym bar example as template.

No RL, just the env running and random actions.
This example is to separate the env from the RL part.

Installation requirements: same as basic lunar (gymnasium[box2d])
'''

import random
import gymnasium as gym

# print(gym.envs.registry.keys())
env = gym.make('CartPole-v1', render_mode='human')
env.reset()

for _ in range(10):
    score = 0
    env.reset()
    while (True):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # len_observation = observation.shape[0]
        # if len_observation < 4:
        #     print('here')
        env.render()
        score = score+reward
        # print(score)
        if terminated or truncated:
            print('terminated' if terminated else 'truncated')
            break

env.close()
