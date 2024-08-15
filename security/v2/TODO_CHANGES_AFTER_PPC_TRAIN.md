PROBLEM 1: attack does not stop when pdc-db is defended too early
- machine3=1 because ability smb has HARDCODED patreides password (so it can execute that ability independent of the other steps success or not)
- attack does not stop probably because adversary tries other machines. Check abilities with the "modified" suffix in the name in the operation
- both problems are not caldera related, the first is not possible in real life, the second is how the adversary sneaks whan an action was not success.
- SOLUTION: either disable caldera ability to sneak (in planner or adversary settings) or give params instead of hardcoded password.

PROBLEM 2: defend machine 1 fails to above when executed the first step
- sometimes [000]->[003] because execute isolate but in the meantime the ssh was executed so detected at the same time. success. THE RL LEARNS 000->ISOLATE
- other times [000]->[000] and execute isolate. Since the attack goes on, this case leads to error later. THE RL FAILS WITH LEARNED 00->ISOLATE

**SOLUTION 1: should be resolved on the attack side, not from here**. Since solution 2 handles that case, we can leave as is for now

**SOLUTION 2: start the attack 1 min after the RL, this way RL learns 000->DO NOTHING and only reward when 100->ISOLATE**



