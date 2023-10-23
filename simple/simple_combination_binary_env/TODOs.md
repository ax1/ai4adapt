# SimpleCombinationEnvBinary [TODOs]

This version changes the original observation_space (1 Dimension float) into a bigger one with more restrictions (N dimensios but binary on each dimension).

Check **WHY** this occurs:
- [ ] a) adding more dimensions let **the policy** for finding better hills and valleys (think 3D gradient mountain analogy)
- [ ] b) or the observation unit is so basic(boolean) that the **hidden neuron layers** activate to the solution very quickly
- [ ] c) or the improvement is due to more obs dimension therefore **more initial input nodes** in the neural network (and in this case the problem indicates that maybe in the DRL algorithms the size and distribution of the neural network is not adapted for minimum structures when the environment or the agent have few "edges")

Check also:
- [ ] why increasing alive-steps-time helps to train 10x faster, compared to default max-env-steps=100 plus tuning the learning_rate in the PPO (giving similar results independent of learning-rate). When increasing the alive-steps there is no need to tune the default learning rate.
