'''
Basic example of built-in environment
Check https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit1/unit1.ipynb
Check also gymnasium_documentation https://gymnasium.farama.org/content/basic_usage/

Installation: pip install gymnasium[box2d]
'''

import gymnasium as gym
import numpy as np
from time import sleep


def execute(observation, reward):
    '''observation: x,y,vx,vy,angle,vangular,left_leg,right_leg
    # action: 0:none, 1:to_left, 2: to_up, 3:to_right, 4: all_engines'''
    print(f'observation: {observation} \nreward: {reward}')
    x, y, vx, vy, a, va, L, R = observation
    action = None
    # Control X
    twist = True if va < -0.20 or va > 0.20 else False

    if x < -0.30 and y < 1 and vy < 0:
        action = np.int64(3)
    elif x > 0.30 and y < 1 and vy < 0:
        action = np.int64(1)

    # Control Angle
    # if a < -0.30 or va < -0.30:
    #     action = np.int64(1)
    # elif a > 0.30 or va > 0.30:
    #     action = np.int64(3)
    angle = abs(x)/2 if abs(x) > 0.2 else 0.1
    if a < -angle:
        action = np.int64(1)
    elif a > angle:
        action = np.int64(3)

    # Default
    if action == None:
        action = np.int64(0)

    return action


env = gym.make("LunarLander-v2", render_mode="human", gravity=-2.0)
# env = gym.make("LunarLander-v2")
env.action_space.seed()
observation, info = env.reset()
initial_action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample())

for _ in range(5000):
    # sleep(0.1)
    action = execute(observation, reward)
    observation, reward, terminated, truncated, info = env.step(action)
    x, y, vx, vy, a, va, L, R = observation
    # for horizontal scrolling turn+up is better than single action
    # if (action == 1 or action == 3) and observation[1] < 0.5:
    if x < -0.2 and y < 0.5:
        if a > -0.05:
            observation, reward, terminated, truncated, info = env.step(
                np.int64(3))
        if vy < 0.5:
            observation, reward, terminated, truncated, info = env.step(
                np.int64(2))

    elif x > 0.2 and y < 0.5:
        if a < 0.05:
            observation, reward, terminated, truncated, info = env.step(
                np.int64(1))
        if vy < 0.5:
            observation, reward, terminated, truncated, info = env.step(
                np.int64(2))

    if terminated or truncated:
        observation, info = env.reset()
    else:
        execute(observation, reward)


env.close()
