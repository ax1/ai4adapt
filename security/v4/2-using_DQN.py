import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from simple_custom_env2 import SimpleCustomEnv

custom_env = SimpleCustomEnv()
# model = DQN("MlpPolicy", custom_env, verbose=1, policy_kwargs=dict(net_arch=[16]), learning_rate=0.001, buffer_size=100, batch_size=8)
# model = DQN("MlpPolicy", custom_env, verbose=1, learning_rate=0.001,
#             train_freq=(5, 'episode'), policy_kwargs=dict(net_arch=[16]))
model = DQN("MlpPolicy", custom_env)
model.learn(total_timesteps=40048, progress_bar=False)

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
# print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()
steps = 0
while (steps < 40):
    steps += 1
    action, _states = model.predict(obs, deterministic=True)
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
