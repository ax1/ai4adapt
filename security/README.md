# SecurityEnvironment (custom Gym env)

Environment representing a security scenario when a set of security mechanisms must be executed to solve a given attack. Since the environment is created as a gym (Gymnasium) environment, we can apply reinforcement learning similar as when solving problems in other scenarios. However, the idea is to have an environment when multiple techniques could be applied, so in the future the env could provide some extensions to allow using novel RL (or non-RL) techniques.

Note: this folder will be refactored once environment is stable:
1- Environment will be packaged as gym env (so it can be imported as with any gym env)
2- Sample code and documentation out of the env

## Installation

```sh
pip install gymnasium[box2d]
```

## Usage

1- Try the lunar folder first, to test the basic env and understand how environment and agents share information
2- Then use the security folder, containing the SecurityEnvironment.py and examples tu use the agent
3- Then extend the SecurityEnvironment as a proxy to the real environments

## How to apply an **well-known** RL strategy

1- Apply into Lunar example and verify
2- Then apply into the SecurityEnvironment

## How to apply an **novel** RL strategy

1- Apply into SecurityEnvironment example and verify
2- Then apply into Lunar, the strategy could be not as efficient as other techniques, but at least, it should resolve the optimization problem. Otherwise, the newly-created strategy is not valid because it should work with any abstract environment (a sort algorithm is not dependent on the items to be sorted, isn't it?)

## Inference
- Use a trained model. Mandatory: the model name should start with the name of the TARGET system as 3 first characters
- Use the files with name "inference"