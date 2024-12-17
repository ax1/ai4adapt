# SimpleCombinationEnv

Environments with a very simple problem wich is hard for PPO to learn even being is a versatile algorithm. Other algorithms based on value learns better these types of problems. 

- Unordered is hard to learn but also check deterministic not to false because it will bang the same action always.
- Ordered is still hard even if less probabilities to find the solution.
- MiniOrdered has only 2 theoretical steps and  PPO still struggles when short of iterations. Ayway the ordered versions once learned are fine with deterministic=true

> Note also that using observation=float or integer limits does not vary the results. The key is the next folder (Combination Binary), where just expanding the observation space, the PPO works better even with the default parameters. **So observation space design, in some cases, is more important than reward strategy or algorithm parameter tuning.**

