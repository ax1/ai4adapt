import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

vec_env = model.get_env()
obs = vec_env.reset()
while (True):
    action, _state = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
