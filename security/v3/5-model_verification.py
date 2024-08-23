'''
(Note that verification is also performed at the end of the training phase)

Model VERIFICATION phase (no training, just runtime)
- Using simulator for faster test
- Test: Model should be load
- Test: Model should recommend valid action based on current state
- Deterministic=False: model should provide 1 or more actions (based on probabilities during the training)
- Mode recommend (action=0) and mode run (action=predict()) can be evaluated here (uncomment lines at the end)
'''
from time import sleep
from security_environment import SecurityEnvironment
from stable_baselines3 import PPO
import os

SIMULATE = True
# MODEL_PATH = '/home/ubuntu/SOFTWARE/AI4CYBER/SOFTWARE/AI4CYBER/AI4ADAPT_REPO/21_08_24_PPC_GOOD/PPC__PPO_656_steps__slot_16__lr1e-3_SB3.zip'
MODEL_PATH = '/home/ubuntu/SOFTWARE/AI4CYBER/SOFTWARE/AI4CYBER/AI4ADAPT_REPO/01_08_2024_DUMMY_ALLWEATHER/HES__PPO_4096_steps__slot_16__SB3.zip'
securityEnvironment = SecurityEnvironment(os.path.basename(MODEL_PATH), simulate=SIMULATE)
print(MODEL_PATH)
model = PPO.load(MODEL_PATH, securityEnvironment)

vec_env = model.get_env()
observations = [securityEnvironment.execute(0)]  # Reset, but do not modify the observation at remote
while True:
    '''
    Deterministic=true gives better results given same training steps.
    Deterministic=true provides probabilistic options
    '''
    sleep(1)
    actions, _states = model.predict(observations, deterministic=False)
    print(f'AI4ADAPT RL agent: based on observations {observations[0]}, recommend executing action {actions[0]}')
    # a-RECOMMEND MODE
    observations = [securityEnvironment.execute(0)]
    # b-EXECUTE MODE
    # observations = [securityEnvironment.execute(actions[0])]
