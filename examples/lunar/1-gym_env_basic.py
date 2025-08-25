'''
Basic example of built-in environment
Check https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit1/unit1.ipynb
Check also gymnasium_documentation https://gymnasium.farama.org/content/basic_usage/

Installation: pip install gymnasium[box2d]
'''

import gymnasium as gym
print(gym.envs.registry.keys())  # print available envs
env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("LunarLander-v2")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(500):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample())
    if terminated or truncated:
        observation, info = env.reset()

env.close()
