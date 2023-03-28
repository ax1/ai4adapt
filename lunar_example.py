'''
#######################
CHANGES, READ FIRST!!!
#######################

- use gymnasium instead of gym
- install virtualscreen as desktop
- gym env.step() parameters from latest version https://gymnasium.farama.org/api/env/


DOCUMENTATION
---------------
https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit1/unit1.ipynb


INSTALLATION
---------------

# swig package to generate bindings between python and c++
sudo apt install swig cmake
# wheel mandatory for box2d
pip install wheel gymnasium[box2d] stable-baselines3[extra] box2d-py pyglet huggingface_sb3

# For rendering
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

# ----------------------GUI----------------------------
from pyvirtualdisplay import Display
disp = Display(visible=True)
disp.start()

# ----------------------ENVIRONMENT--------------------
env = gym.make("LunarLander-v2")
observation = env.reset()
for _ in range(20):
    action = env.action_space.sample()
    print("Action taken:", action)
    observation, reward, terminated, truncated, info = env.step(action)
    # args = env.step(action)
    if terminated or truncated:
        print("Environment is reset")
        observation = env.reset()
