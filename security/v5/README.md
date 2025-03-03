# Improve do-nothing at expense of less total effectiveness

## Analysis for V2

In v2 reward system, we set time reward=0 because we already reward to resolve as quick as possible by incremental rewarding/penalty until solution.
- Advantage: we ensure that lazy defenses are discarded and the best skilled defenses are the ones to execute as fast as possible, because the more steps, any failed defense tracks its penalty and another penalty from others for not using its step to resolve the problem. With 1024 steps we can train the RL agent even when using lots of defenses.
- Disadvantage: do-nothing has a little reward but it is hard to learn to use it because it has the previous penalties from others. we tested some tweaks to improve this but we still need 3 or 4 times more iterations to learn this. In the real system training time is a very big constraint. 

## Approach for V3

If we go for the classical reward approach, we will miss the very efficient defenses sometimes, but the agent should learn to stand still when no action is required.

## Procedure

- **reward atomically (usually the default)**. We will miss the superfast learning when a defense is success, because previously we had `r + sum r + success` and now we only have `r + success` and the useless defenses were discarded faster(penalties) `r + sum r` where now is only `r`.
- **add penalty for currently unused reward variables related to time**. This will help to increase the learning of not only what to do, but also when is better. The time rewards/penalty was before as implicit reward because on each step we already gave that bonus. Now we need to explicit reward time alive (or penalty in case we prefer to foster resolve asap).
- **keep reward on do-nothing, but give only a tiny reward.** We want to learn to wait, but wait **is not** a good defense so promote always execute defenses in the first steps of the training to find quickly some of the good ones (example remember DQN vs PPO folder where for n iterations DQN was better but solution was learn to fly and not try to land, so in the end we needed more iterations and in that case PPO was much better). For spare iterations, the min reward will take effect, but for few iterations we should still have faster promotion of good defenses compared to do-nothing.
- **try back to algorithm defaults or small tuning for convergence**. Now a good defense is much higher than a valid or delayed defense, so we do not need to "force" the algorithm to learn that defense. We also avoid other RL algorithms that are fit for learn asap but the are not very good for refined behaviour.

## Potential problems before implementing V3

- The env is a real system, sometimes valid defenses are not executed properly and they are missed until a next iteration resolves success. In case of very unreliable environment, this version will behave extremely poor compared to v2. So we need to test environment stability first.
- We need to tune carefully many rewards. Previously, we only needed to give penalties to all defenses until a good one, then give a lot of reward. Now we must be careful about reward picking because it can lead to very different strategies for the RL agent. For example, small penalties will lead to lazy agent. High time penalties will lead to RL being too preventive instead of executing based on current situation, or if we will gift for staying alive time, the RL will learn too quickly to save only the final machine and the rest of the time been idle.
- Go back to V2 is no good results here. We already know, not only by security training but also from simpleBinary that incremental reward makes go straight to the solution even if the path is not optimal, so we can use only one algorithm for many cases including the ones where HER for goal based or SAC for envs with sparse rewards are, in theory, better, or the most generalistic DQN. PPO can improve iteratively when the env is very complex and does not trap into partial solutions. 
