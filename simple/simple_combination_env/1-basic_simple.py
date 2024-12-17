from SimpleCombinationEnvOrdered import SimpleEnv
import numpy as np

env = SimpleEnv()
actions = list(range(env.ACTIONS))
valids = []
observation, info = env.reset()
last = observation[0]
for r in range(100):
    action = np.random.choice(actions)
    observation, reward, terminated, truncated, info = env.step(action)
    obs = observation[0].item()
    if last < obs:
        last = observation[0].item()
        valids.append(action)
        actions.remove(action)
    print(f'action {action} Reward: {reward} next_observation {observation}')
    if terminated or truncated:
        print(info)
        if terminated:
            actions = valids
        env.reset()
        valids = []
        last = 0
