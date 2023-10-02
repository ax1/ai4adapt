'''
Example using LUNAR + DQN + SB3

Note: check also the lunar PPO to compare (same degree of skill)
- number of iterations
- total time to learn
- convergence speed to solution
- overfitting (by looking at glitches on some episodes)

Results:
- DQN 250K (same with 100K) iterations still not optimal solution (learns to hover but do not land)
'''

import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

ENV_NAME = "LunarLander-v2"
LEARN_ITERATIONS = 25_000
MODEL_NAME = ENV_NAME+'_DQN_'+str(LEARN_ITERATIONS)
# Create environment
env = gym.make(ENV_NAME, render_mode="rgb_array")

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=LEARN_ITERATIONS, progress_bar=True)
# Save the agent
model.save(MODEL_NAME)
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load(MODEL_NAME, env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
while (True):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
