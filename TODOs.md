# TODOs
- [ ] Reduce time. Unless a script to have all the actions (so we can test defenses are ok). Then, in caldera operations either execute the script with args 8eg: 1 4 8 execute only those in the script) or execute loop actions with an argument on each execution (to prevent create operations for all the steps and also to handle to reduce the time of each episode for training). NOT POSSIBLE FOR NOW.

- [x] check why model save error when not registered Solved, load constructor has an option to add an instance of the env instead of registered
- **[x] TOP PRIORITY In wazuh or in adapt proxmox, print also the timestamps of the action and the observations and execute the ONLY DOROTHY (2 hours 128 steps) and send report to Eider** Put timestamp epoch in seconds in adapt report and also in adapt proxmox
- [x] In wazuh  in the mitigations_resolved function, also check attack resolve by looking at 300 220 200 etc (not only 333)
- [x]  In wazuh in  when the attack cannot go anywhere else, also stop spider wizard and notify in the return
-[x] in adapt proxm, after wazu sensing with resolved and wizard stopped, send observation and also the stop signal to adapt_environment eg sending 333 or sending additional data in the clipboard. GHANGED: send obs only, the success based on observ is done in the sec_env step()
- [x] in adapt-env, send back the termination of the experiment (attack cannot go further). Same as above, end signal can be derived from observation so no need to add signal
- [x] in rl-agent , truncate is either FUU WIN or ATTACK FINISHED SIGNAL (terminate is still as is 50 max steps) 
- [x] in rl-agent change win strategy to a matrix table [X00, XX0]  