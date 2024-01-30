# @@@@@@@@@@@@@ TODOO SWITCH BACK TO SEC_ENV NOT DUMMY
from dorothy_security_environment import SecurityEnvironment
from stable_baselines3 import PPO

MAX_TRAINING_STEPS = 1_00
EXPERIMENT = f'PPO {MAX_TRAINING_STEPS} steps, default params, SB3'

def train():
    # lookout: PPO default block steps aways forced to 2048 blocks, override with n_steps
    # model = PPO("MlpPolicy", SecurityEnvironment(), verbose=1, learning_rate=0.1, gamma=0.01)
    model = PPO("MlpPolicy", SecurityEnvironment(EXPERIMENT), verbose=1, n_steps=128)
    model.learn(total_timesteps=MAX_TRAINING_STEPS, progress_bar=False)
    return model


def test(model):
    print('\033[94m')
    vec_env = model.get_env()
    observations = vec_env.reset()
    for r in range(50):
        '''
        Deterministic=true gives better results given same training steps.
        '''
        actions, _states = model.predict(observations, deterministic=True)
        observations, rewards, dones, info = vec_env.step(actions)
        # if dones:
        # print(info)
        # vec_env.reset()


# model = train()
# model.save(EXPERIMENT.replace(',', '_').replace(' ', '_'))
model = PPO.load('PPO_100_steps__default_params__SB3')
test(model)
