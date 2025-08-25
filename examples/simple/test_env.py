from stable_baselines3.common.env_checker import check_env as check_env_sb3
from gymnasium.utils.env_checker import check_env as check_env_gym
from SimpleEnv import SimpleEnv


def main():
    print('---Testing environment...---')
    env = SimpleEnv()
    check_env_gym(env)
    check_env_sb3(env)
    print('----------End test----------')


if __name__ == "__main__":
    main()
