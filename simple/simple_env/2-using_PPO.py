from SimpleEnv import SimpleEnv
from stable_baselines3 import PPO

model = PPO("MlpPolicy", SimpleEnv(), verbose=1)
model.learn(total_timesteps=10000, progress_bar=True)
vec_env = model.get_env()
observations = vec_env.reset()
for r in range(1000):
    '''
    Deterministic=true gives better results given same training steps.
    '''
    actions, _states = model.predict(observations, deterministic=True)
    observations, rewards, dones, info = vec_env.step(actions)
    # print(f'observation: {observation}')
    # print(f'Reward: {reward}')
    if dones:
        print(actions)
        print(f'Rewards: {rewards}')
        print(info)
        vec_env.reset()
