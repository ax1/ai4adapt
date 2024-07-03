'''
Start the agent in run mode (no training)

The model name provided in the program input should have the same action_space and
observation_space in the environment application.

'''
import sys
from time import sleep

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
securityEnvironment = SecurityEnvironment(MODEL)

# Tune max steps on recommendation mode
securityEnvironment.MAX_STEPS = 1e6

print(f'{dirname}/{MODEL_FILE}')
model = PPO.load(f'{dirname}/{MODEL_FILE}', securityEnvironment)

vec_env = model.get_env()
observations = vec_env.reset()
counter_episodes = 0

while True:
    '''
    Deterministic=true gives better results given same training steps.
    '''
    sleep(10)
    actions, _states = model.predict(observations, deterministic=True)
    print(f'AIADAPT RL agent: based on observations {observations[0]}, recommend executing action {actions[0]}')
    observations, rewards, dones, info = vec_env.step(actions)
    if dones:
        counter_episodes += 1
        # print(info)
        # vec_env.reset()
