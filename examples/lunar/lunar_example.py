'''
Example of LUNAR, with focus on simpler env understanding and different ways (custom) to RL train the model 

@@@@@@@@@@@@@@@@@@@@@@@@
CHANGES, READ FIRST!!!
@@@@@@@@@@@@@@@@@@@@@@@@

- use gymnasium instead of gym
- [CHECK if now we can remove all dependencies on this (pip & apt)] install virtualscreen as desktop [I think vscreen is optional because see my other bare gym exammple in the same folder]
- gym env.step() parameters from latest version https://gymnasium.farama.org/api/env/
- use render_mode human in the env creation to display


DOCUMENTATION
---------------
https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit1/unit1.ipynb


INSTALLATION
---------------

# swig package to generate bindings between python and c++
sudo apt install swig cmake
# wheel mandatory for box2d
pip install wheel gymnasium[box2d] stable-baselines3[extra] box2d-py pyglet huggingface_sb3

# For rendering [REMINDER: vdisplay dependencies could be not needed now thanks to 'human' parameter in env creation]
 sudo apt install python-opengl ffmpeg ffmpeg
pip install pyvirtualdisplay

# If problems with virtualdisplay install as follows here https://github.com/ponty/pyvirtualdisplay/tree/3.0
'''

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
# from huggingface_hub import notebook_login
# from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
import gymnasium as gym

# ----------------------ENVIRONMENT--------------------
'''Demonstration of gattin g parameters from the environment'''
env = gym.make("LunarLander-v2",)  # headless mode
env = gym.make("LunarLander-v2", render_mode="human")  # UI mode
observation = env.reset()

print("Observation Space Shape (vector size)", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())

print("Action Space Shape (vector size)", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action

# ---------------------CUSTOM TRAINING------------------
'''
Iterate on the environment and apply any custom or ad-hocs formulas for training (no specific algorithm is required)
The idea is to demonstrate that we only need to create a custom 'SOAR' env and then we can apply our own strategy for rewards (or apply generic algorithms (probablilistic, etc) instead of the most used for RL,)
'''
for _ in range(500):
    action = env.action_space.sample()
    print("Action taken:", action)
    observation, reward, terminated, truncated, info = env.step(action)
    # args = env.step(action)
    if terminated or truncated:
        print("Environment is reset")
        observation = env.reset()

# -------------------NEURAL MODE TRAINING----------------
'''ERROR->SOLUTION: After googleing, the problem with stable_baseline algorithms is that do not so compatible with newer gym (or any gymnasium) version so install an older version of gym (not gymnasyum, to run this training) https://stackoverflow.com/questions/75108957/assertionerror-the-algorithm-only-supports-class-gym-spaces-box-box-as-acti'''
env = gym.make('LunarLander-v2')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(2e4))
