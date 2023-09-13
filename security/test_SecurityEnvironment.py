from security_environment import SecurityEnvironment
import gymnasium as gym
ENV_NAME = 'sec_env'


def test_register():
    SecurityEnvironment.register(ENV_NAME)


def test_use_registered():
    env = gym.make(ENV_NAME)


def test_use_normal():
    env = SecurityEnvironment()


test_use_normal()
test_register()
test_use_registered()
