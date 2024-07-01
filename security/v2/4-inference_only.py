'''
Start the agent in run mode (no training)

The model name provided in the program input should have the same action_space and
observation_space in the environment application.

'''

import sys
if len(sys.argv) < 2:
    print("""
          Usage: python inference.py $model_name
            (where model name format is "$TARGET $ALGORITHM $PARAMS")
          """)
    exit(1)


from security_environment import SecurityEnvironment
from stable_baselines3 import PPO
import os


MODEL = sys.argv[1]
MODEL_FILE = MODEL.replace(',', '_').replace(' ', '_')
dirname, filename = os.path.split(os.path.abspath(__file__))
model = PPO.load(f'{dirname}/{MODEL_FILE}', SecurityEnvironment(MODEL))
vec_env = model.get_env()
observations = vec_env.reset()
counter_episodes = 0
while counter_episodes < 5:
    '''
    Deterministic=true gives better results given same training steps.
    '''
    actions, _states = model.predict(observations, deterministic=True)
    observations, rewards, dones, info = vec_env.step(actions)
    if dones:
        counter_episodes += 1
        # print(info)
        # vec_env.reset()
