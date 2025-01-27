from security_environment import SecurityEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

TARGET = 'PPC'  # TARGET IMPORTANT !!! (3 characters)
SIMULATE = False

MAX_TRAINING_STEPS = 1024
TRAIN_SLOT = 16
MODEL = f'{TARGET}, PPO {MAX_TRAINING_STEPS} steps, slot {TRAIN_SLOT}, SB3'
MODEL_FILE = MODEL.replace(',', '_').replace(' ', '_')


def train():

    # Lookout: PPO default block steps aways forced to 2048 blocks, override with n_steps
    # model = PPO("MlpPolicy", SecurityEnvironment(), verbose=1, learning_rate=0.1, gamma=0.01)
    '''
    ON TRAINING PAY ATTENTION to ep_rew_mean at rollout info (eg: our success is 100 less 1 defense so anywhere near 87-90 is good training)

    KEEP TRAIN_SLOT TO SMALL VALUES because in the real system we can spend time on updating policies and our episodes are short

    DO NOT REMOVE n_epochs, in SB3 default is 10 but for finding partial solutions faster, 50 this is giving
    much better results than tweaking the learning_rate. Epochs increase the number of times that
    the policy is update during a slot of n_steps. Either increase epoch given many steps for finding the best
    solution or use less epoch and less n_steps, to find partial solutions faster.
    https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    https://spinningup.openai.com/en/latest/algorithms/ppo.html
    '''
    print('\n---------------TRAIN MODEL--------------------')
    # Save model from time to time https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
    save_callback = CheckpointCallback(save_freq=TRAIN_SLOT, save_path="./temp/",
                                       save_replay_buffer=True, save_vecnormalize=True)

    # Train the agent to defend the environment
    securityEnvironment = SecurityEnvironment(MODEL_FILE, simulate=SIMULATE, atomic=True)
    model = PPO("MlpPolicy", securityEnvironment,
                verbose=1, n_epochs=50, n_steps=TRAIN_SLOT, batch_size=TRAIN_SLOT, learning_rate=0.001)

    # Train it with most of the default options (only cap the ones related to train size)
    # model = PPO("MlpPolicy", securityEnvironment, verbose=1)

    # Try fostering dqn vs ppo but params ent_coef = 1 to foster exploration or vf coef to foster value results does not work with simulator so far
    # model = PPO("MlpPolicy", securityEnvironment,
    #            verbose=1, n_epochs=50, n_steps=TRAIN_SLOT, batch_size=TRAIN_SLOT, learning_rate=0.001, vf_coef=1, ent_coef=1)

    # Print raw parameters info
    print(model.__dict__)

    model.learn(total_timesteps=MAX_TRAINING_STEPS, progress_bar=False, callback=save_callback)
    return model


def test(model):
    print('\n---------------TEST MODEL--------------------')
    vec_env = model.get_env()
    observations = vec_env.reset()
    counter_episodes = 0
    while counter_episodes < 10:
        '''
        Deterministic=true gives better results given same training steps.
        '''
        actions, _states = model.predict(observations, deterministic=True)
        observations, rewards, dones, info = vec_env.step(actions)
        if dones:
            counter_episodes += 1
            # print(info)
            # vec_env.reset()


# ---TRAIN---
model = train()
model.save(MODEL_FILE)

# ---TEST---
# dirname, filename = os.path.split(os.path.abspath(__file__))
# model = PPO.load(f'{dirname}/{MODEL_FILE}', SecurityEnvironment(MODEL))
test(model)
