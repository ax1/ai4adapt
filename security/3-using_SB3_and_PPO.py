import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from security_environment import *

ENV_NAME = "SecurityEnvironment"
LEARN_ITERATIONS = 1000
MODEL_NAME = ENV_NAME+'_'+str(LEARN_ITERATIONS)

SecurityEnvironment.register(ENV_NAME)

vec_env = make_vec_env(ENV_NAME, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=LEARN_ITERATIONS)
model.save(MODEL_NAME)

# del model  # remove to demonstrate saving and loading

model = PPO.load(MODEL_NAME)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    # TODO ARF: with newer envs, done is replaced by terminated truncated
    obs, rewards, dones, info = vec_env.step(action)
#    vec_env.render("human")
