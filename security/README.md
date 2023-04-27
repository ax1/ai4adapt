# SecurityEnvironment (custom Gym env)

Enviroment represeitning a security scenario when a set of security mechanisms must be executed to solve a given attack. Since the environment is created as as gym (Gymnasium) enviroment, we can apply reinforcment learning similar as when solving problems in other scenarios. However, the idea is to have an enviromnet when multiple techniques could be applied, so in the future the env could have some extensions to allow using novel RL techniques.

Note: this folder will be refactored once environment is stable:
1- Environment will be packaged as gym env
2- Sample code and documentation out of the env

## How to apply an **well-known** RL strategy
1- Apply into Lunar example and verify
2- Then apply into the SecurityEnvironment

## How to apply an **novel** RL strategy
1- Apply into SecurityEnvironment example and verify
2- Then apply into Lunar, the strategy could be not as efficient as other existing ones, but at least, it should resolve the optimization problem. Otherwise, the newly-created strategy is not valid because it should work on any abstract environment