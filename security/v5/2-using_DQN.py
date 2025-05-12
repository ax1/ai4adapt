import gymnasium as gym
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from security_environment import SecurityEnvironment
from stable_baselines3.common.callbacks import CheckpointCallback

TARGET = 'PPC'  # TARGET IMPORTANT !!! (3 characters)
SIMULATE = True

MAX_TRAINING_STEPS = 512
TRAIN_SLOT = 256  # in real envs we have low steps, we can afford bigger minibatches
MODEL = f'{TARGET}, DQN {MAX_TRAINING_STEPS} steps, slot {TRAIN_SLOT}, SB3,{datetime.now().strftime("%Y%m%d_%H%M%S")}'
MODEL_FILE = MODEL.replace(',', '_').replace(' ', '_')


def train():
    print('\n---------------TRAIN MODEL--------------------')
    # Save model from time to time https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
    save_callback = CheckpointCallback(save_freq=TRAIN_SLOT, save_path="./temp/",
                                       save_replay_buffer=True, save_vecnormalize=True)
    securityEnvironment = SecurityEnvironment(MODEL_FILE, simulate=SIMULATE, atomic=True)
    '''
    [NO, with 10 atomic reward we can also do it but wit 50 both ppo and dqn performs better] Increase atomic reward to 50: to promote do_nothing/defense finding vs terminal state reward (previously there were few [000]donothing[100] and now we have lots)
    [NO,achieve also low loss with default] Increase minibatch size: because we do not have cpu constraints
    [YES, it is too big compared to ppo best, but it gives massive convergence] Increase a lot the learning rate: force quickly to set good/bad defenses so we reduce the action_space combinatory
    [NO, we also achieve low loss with default, so not required to tune] Reduce gamma: we foster atomic findings versus future reward
    [YES, not as critical as LR but it improves total loss] Gradient_steps: yes we do the same as n_steps in ppo because we are plenty of cpu/gpu time in real envs
    [NO, we improve, but not as much as other params] Increase a little exploration_final: we help to keep exploring, but not much, otherwise we will not find complete trajectories, only atomic positives.
    
    >>>IMPORTANT:<<< So for our problem, HIGH increase of the LR and increasing the gradient steps are the relevant parameter for convergence, the others are only helpers
    '''
    # THIS: dummy changed update to 7 to have more 000 at early training and foster learning do-nothing at the beginning
    # THIS: increased also the reward for do nothing at the same level of valid_defense (SAVE_BULLETS=50, VALID_DEFENSE=50)this also marks do_nothing and good at the beginning of trajectories
    # added also reward bullets same as valid defense to force all trajectories having 000 have big rewards only by optimizing do nothing
    # SO now we have a lot of EEE as we wanted
    # exploration_final_eps: to keep exploring. It cannot be high otherwise it does not give the ones with terminal state as best
    # model = DQN("MlpPolicy", securityEnvironment, verbose=1)
    model = DQN("MlpPolicy", securityEnvironment, verbose=1, learning_rate=0.01, gradient_steps=50)
    # model = DQN("MlpPolicy", securityEnvironment, verbose=1, learning_rate=0.01, gradient_steps=50, exploration_initial_eps=1.0, exploration_final_eps=0.1, gamma=0.99, batch_size=TRAIN_SLOT)  # low expl increase continuous random (help to exec do-nothings but also reduce good trajectories)
    # model = DQN("MlpPolicy", securityEnvironment, verbose=1, learning_rate=0.001, target_update_interval=100, batch_size=TRAIN_SLOT)
    # model = DQN("MlpPolicy", securityEnvironment, verbose=1,
    #             batch_size=TRAIN_SLOT, gamma=0.99, exploration_fraction=0.005)  # final eps to force explore
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
