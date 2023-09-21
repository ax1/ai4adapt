import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from SimpleSliderEnv import SimpleEnv

env = SimpleEnv()

# ENV_NAME = "LunarLander-v2"
# LEARN_ITERATIONS = 25_000
# MODEL_NAME = ENV_NAME+'_DQN_'+str(LEARN_ITERATIONS)
# # Create environment
# env = gym.make(ENV_NAME, render_mode="rgb_array")

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=10000, progress_bar=True)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for r in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones:
        print(f'action {action} Reward: {rewards} next_observation {obs}')
        print(info)
        env.reset()
