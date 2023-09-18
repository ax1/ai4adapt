import gymnasium as gym
from SimpleEnv import SimpleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_NAME = "SimpleEnv"
LEARN_ITERATIONS = 10000
MODEL_NAME = ENV_NAME+'_PPO_'+str(LEARN_ITERATIONS)


vec_env = make_vec_env(SimpleEnv, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=LEARN_ITERATIONS, progress_bar=True)

observations = vec_env.reset()
for r in range(1000):
    actions, _states = model.predict(observations)
    observations, rewards, dones, info = vec_env.step(actions)
    # print(f'observation: {observation}')
    # print(f'Reward: {reward}')
    if dones:
        print(f'Rewards: {rewards}')
        # print(info)
        vec_env.reset()
