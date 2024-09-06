# SimpleEnvBinary

This is a variant of SimpleCombination but **expanding the observation space** to more parameters of less size.

Note that **this improves a lot the SimpleCombinationEnv problem** (this binary version is even the unordered version of simple combination, so the ordered would be even better), and the env performs much better (less than 300 steps vs 4096) even with the default parameters in the algorithm. Apparently, given the same neural network default size, having more inputs to the network (10 booleans) is better than one input of 1 integer between 0-10).

Note also that atomic mode is slightly better than cumulative, but depending on the complexity or on the episode time, the cumulative could perform better by adding an incremental penalty on each step wasted.