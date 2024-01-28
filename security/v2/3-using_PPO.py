from security_environment import SecurityEnvironment
from stable_baselines3 import PPO


def train():
    # lookout: PPO default block steps aways forced to 2048 blocks, override with n_steps
    # model = PPO("MlpPolicy", SecurityEnvironment(), verbose=1, learning_rate=0.1, gamma=0.01)
    model = PPO("MlpPolicy", SecurityEnvironment(), verbose=1, n_steps=256)
    model.learn(total_timesteps=1_000, progress_bar=False)
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


model = train()
# model.save('SEC_ENV')
test(model)
