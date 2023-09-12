The discrete error description is here https://stackoverflow.com/questions/75108957/assertionerror-the-algorithm-only-supports-class-gym-spaces-box-box-as-acti
But it was my fault (I had an old version of SB3 (1.7), after reinstalling the newer everything ok).

also here simple example with PPO https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

- env response : SB3 examples use done, but not sure if done is returned as array by the make_vec_env itself because latest versions of gymnasium uses terminated, truncated instead of done.
