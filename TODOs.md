# TODOs

- [x] Multi-tenant for production, send the agent name to the adapt_env service. For now, since modes are different better to use several instances of the same program. The multi is already done in the env, so there are different URLS
- [ ] Reduce time. Unless a script to have all the actions (so we can test defenses are ok). Then, in Caldera operations either execute the script with args (eg: 1 4 8 execute only those in the script) or execute loop actions with an argument on each execution (to prevent create operations for all the steps and also to handle to reduce the time of each episode for training). NOT POSSIBLE FOR NOW.

- [x] Model check if we can model.save() and the continue with model.load()+model.train() 
- [ ] Use only one 0-do nothing action, also reward positively that action instead of penalty on the other actions and compare if better results
- [ ] Put same report and model name to prevent overwriting the model on each background training
- [ ] instead of 13*actions, create two actions back forward to change the SAME action into different target, this reduces the number of total actions (think platforms game eg: mario bros)
- [ ] 04-07-2024 - Test this: for the real env (or test first with our crafted simulators for security), test any off-policy algorithm eg SAC, DQN since we have very small observation space, and we lose some of the advantages that PPO can offer (DQN we tested in general-purpose envs and did not work better than PPO but for small obs spaces could also worth).