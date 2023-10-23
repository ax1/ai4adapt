'''
LOOKOUT with CPU speed settings

This demo works SUCCESS with 1000 train iterations if power settings=performance(100%)
However this partial results and therefore truncate with same train iterations and power=quiet(30%)
Tested when SimpleEnvMaxSteps=1000 and learn_timesteps=1000
Other tests tuning learn_rate and discount do not have better results for now
'''

from SimpleCombinationBinaryEnv import SimpleEnv
from stable_baselines3 import PPO

# lookout: PPO default block steps aways forced to 2048 blocks, override with n_steps
# model = PPO("MlpPolicy", SimpleEnv(), verbose=1, learning_rate=0.1, gamma=0.01)
model = PPO("MlpPolicy", SimpleEnv(), verbose=1, n_steps=128)
model.learn(total_timesteps=200, progress_bar=True)
print(f'>>> Total steps so far: {model.num_timesteps}')

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
    if dones:
        print(info)
        # vec_env.reset()
