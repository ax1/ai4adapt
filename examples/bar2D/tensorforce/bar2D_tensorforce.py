'''
See https://github.com/tensorforce/tensorforce

Installation: pip instal tensorforce (it takes several minutes)

Run: error TypeError: Descriptors cannot not be created directly. Request to downgrade protobuf installed is 4.x and required is 3.x or less (whic is a bad ide)

Summary: it requires downgrading a lib, and innstallation appears to download a lot. Nevertheless, the code is clean 

'''

from tensorforce import Agent, Environment

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    # alternatively: states, actions, (max_episode_timesteps)
    environment=environment,
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

# Train for 300 episodes
for _ in range(300):

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()
