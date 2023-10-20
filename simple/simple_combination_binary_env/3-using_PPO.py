'''
LOOKOUT with CPU speed

This demo works SUCCESS with 1000 train iterations if power settings=performance(100%)
However this partial results and therefore truncate with same train iterations and power=quiet(30%)
'''

from SimpleCombinationEnvBinary import SimpleEnv
from stable_baselines3 import PPO

model = PPO("MlpPolicy", SimpleEnv(), verbose=1)
model.learn(total_timesteps=1_000, progress_bar=True)
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
        print(info)
        # vec_env.reset()
