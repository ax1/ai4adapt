# Test 1024 and env starts from 000

- default 1024 env from 000: learns save dorothy but not do nothing
- same but save_bullets=20: learns both save doro and do_nothing
- save_bullets=20: and incremental reward: learns both save doro and do_nothing

...so incremental or atomic does not change convergence very much