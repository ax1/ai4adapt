from security_environment import SecurityEnvironment
from stable_baselines3 import PPO
import os


MAX_TRAINING_STEPS = 2048
MODEL = f'PPO {MAX_TRAINING_STEPS} steps, default params, SB3'
MODEL_FILE = MODEL.replace(',', '_').replace(' ', '_')

def train():
    # lookout: PPO default block steps aways forced to 2048 blocks, override with n_steps
    # model = PPO("MlpPolicy", SecurityEnvironment(), verbose=1, learning_rate=0.1, gamma=0.01)
    model = PPO("MlpPolicy", SecurityEnvironment(MODEL_FILE), verbose=1, n_steps=MAX_TRAINING_STEPS)
    model.learn(total_timesteps=MAX_TRAINING_STEPS, progress_bar=False)
    return model


def test(model):
    print('\033[94m')
    vec_env = model.get_env()
    observations = vec_env.reset()
    for r in range(200):
        '''
        Deterministic=true gives better results given same training steps.
        '''
        actions, _states = model.predict(observations, deterministic=True)
        observations, rewards, dones, info = vec_env.step(actions)
        # if dones:
        # print(info)
        # vec_env.reset()


# ---TRAIN---
model = train()
# model.save(MODEL_FILE)

# ---TEST---
# dirname, filename = os.path.split(os.path.abspath(__file__))
# model = PPO.load(f'{dirname}/{MODEL_FILE}', SecurityEnvironment(MODEL))
test(model)
