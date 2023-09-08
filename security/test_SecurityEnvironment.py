from SecurityEnvironment import *
import gymnasium as gym
ENV_NAME = 'sec_env'


def test_register():
    SecurityEnvironment.register(ENV_NAME)


def test_use_registered():
    env = gym.make(ENV_NAME)


test_register()
test_use_registered()
