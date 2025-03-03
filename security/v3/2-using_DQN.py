import gymnasium as gym
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from security_environment import SecurityEnvironment

TARGET = 'PPC'  # TARGET IMPORTANT !!! (3 characters)
SIMULATE = True

MAX_TRAINING_STEPS = 4048
TRAIN_SLOT = 16
MODEL = f'{TARGET}, DQN {MAX_TRAINING_STEPS} steps, slot {TRAIN_SLOT}, SB3,{datetime.now().strftime("%Y%m%d_%H%M%S")}'
MODEL_FILE = MODEL.replace(',', '_').replace(' ', '_')

securityEnvironment = SecurityEnvironment(MODEL_FILE, simulate=SIMULATE, atomic=True)
model = DQN("MlpPolicy", securityEnvironment, verbose=1, learning_rate=0.01,
            target_update_interval=100, batch_size=TRAIN_SLOT)
# model = DQN("MlpPolicy", securityEnvironment, verbose=1)
model.learn(total_timesteps=MAX_TRAINING_STEPS, progress_bar=True)
# model = DQN("MlpPolicy", custom_env, verbose=1, policy_kwargs=dict(net_arch=[16]), learning_rate=0.001, buffer_size=100, batch_size=8)
# model = DQN("MlpPolicy", custom_env, verbose=1, learning_rate=0.001,
#             train_freq=(5, 'episode'), policy_kwargs=dict(net_arch=[16]))

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
# print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()
steps = 0
while (steps < 100):
    steps += 1
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    print(action, obs, rewards, dones)


'''
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
# from security_environment import SecurityEnvironment
import SimpleCustomEnv

# securityEnvironment = SecurityEnvironment('model_dqn', simulate=True, atomic=True)
securityEnvironment = SimpleCustomEnv()
model = DQN("MlpPolicy", securityEnvironment, verbose=1)
model.learn(total_timesteps=2024, progress_bar=True)

print('test')
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)

# vec_env = model.get_env()
# obs = vec_env.reset()
# steps = 0
# while (steps < 40):
#     steps += 1
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
'''
