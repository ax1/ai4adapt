# TODOs
- [ ] Reduce time. Unless a script to have all the actions (so we can test defenses are ok). Then, in caldera operations either execute the script with args 8eg: 1 4 8 execute only those in the script) or execute loop actions with an argument on each execution (to prevent create operations for all the steps and also to handle to reduce the time of each episode for training). NOT POSSIBLE FOR NOW.

- **[ ] TOP PRIORITY In wazuh or in adapt proxmox, print also the timestamps of the action and the observations and execute the ONLY DOROTHY (2 hours 128 steps) and send report to Eider** Put timestamp epoch in seconds in adapt report and also in adapt proxmox
- [ ] In wazuh  in the mitigations_resolved function, also check attack resolve by looking at 300 220 200 etc (not only 333)
- [ ]  In wazuh in  when the attack cannot go anywhere else, also stop spider wizard and notify in the return
in adapt proxm, after wazu sensing with resolved and wizard stooped, send observation and also the stop signal to adapt_environment eg sending 333 or sending additional data in the clipboard
- [ ] in adapt-env, send back the termination of the experiment (attack cannot go further)
- [ ] in rl-agent , truncate is either FUU WIN or ATTACK FINISHED SIGNAL (terminate is still as is 50 max steps) 
- [ ] in rl-agent change win strategy to a matrix table [X00, XX0]  