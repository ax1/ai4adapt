from SimpleCombinationEnvMiniOrdered import SimpleEnv
from stable_baselines3 import DQN

model = DQN("MlpPolicy", SimpleEnv(), verbose=1)
model.learn(total_timesteps=1_000, progress_bar=True)
vec_env = model.get_env()
observations = vec_env.reset()
for r in range(100):
    '''
    Deterministic=true gives better results given same training steps.
    '''
    actions, _states = model.predict(observations, deterministic=True)
    observations, rewards, dones, info = vec_env.step(actions)
    # print(f'observation: {observation}')
    # print(f'Reward: {reward}')
