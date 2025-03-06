import gymnasium as gym
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from security_environment import SecurityEnvironment
from stable_baselines3.common.callbacks import CheckpointCallback

TARGET = 'PPC'  # TARGET IMPORTANT !!! (3 characters)
SIMULATE = False

MAX_TRAINING_STEPS = 2048
TRAIN_SLOT = 64
MODEL = f'{TARGET}, DQN {MAX_TRAINING_STEPS} steps, slot {TRAIN_SLOT}, SB3,{datetime.now().strftime("%Y%m%d_%H%M%S")}'
MODEL_FILE = MODEL.replace(',', '_').replace(' ', '_')


def train():
    print('\n---------------TRAIN MODEL--------------------')
    # Save model from time to time https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
    save_callback = CheckpointCallback(save_freq=TRAIN_SLOT, save_path="./temp/",
                                       save_replay_buffer=True, save_vecnormalize=True)
    securityEnvironment = SecurityEnvironment(MODEL_FILE, simulate=SIMULATE, atomic=True)
    # model = DQN("MlpPolicy", securityEnvironment, verbose=1, learning_rate=0.001, target_update_interval=100, batch_size=TRAIN_SLOT)
    model = DQN("MlpPolicy", securityEnvironment, verbose=1, exploration_final_eps=0.5,
                batch_size=TRAIN_SLOT)  # final eps to force explore
    print(model.__dict__)
    model.learn(total_timesteps=MAX_TRAINING_STEPS, progress_bar=False, callback=save_callback)
    return model


def test(model):
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
    # print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    print('\n---------------TEST MODEL--------------------')
    vec_env = model.get_env()
    observations = vec_env.reset()
    counter_episodes = 0
    while counter_episodes < 20:
        actions, _states = model.predict(observations, deterministic=True)
        observations, rewards, dones, info = vec_env.step(actions)
        if dones:
            counter_episodes += 1


# ---TRAIN---
model = train()
model.save(MODEL_FILE)

# ---TEST---
test(model)
