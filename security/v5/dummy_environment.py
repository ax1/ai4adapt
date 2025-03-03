import random

# The BIG one
# ACTIONS = [{"pos": 0, "name": "Do nothing (leave system evolving)", "ability_id": "", "target": "dorothy"}, {"pos": 1, "name": "Disable user account", "ability_id": "ad0b71fe-1b2d-4847-a24a-f4de322ac360", "target": "dorothy"}, {"pos": 2, "name": "Force password reset", "ability_id": "4e5c5024-765d-4ff5-ae68-6d1d496c8bef", "target": "dorothy"}, {"pos": 3, "name": "Force sign out", "ability_id": "9fd2778f-91a9-469f-8357-0cb50b02a4ae", "target": "dorothy"}, {"pos": 4, "name": "Add to block list", "ability_id": "67741162-0d58-4021-bb60-0da5462db181", "target": "dorothy"}, {"pos": 5, "name": "Isolate endpoint", "ability_id": "040db47b-1a7b-439e-ae78-a80d5d5c5812", "target": "dorothy"}, {"pos": 6, "name": "Terminate process", "ability_id": "a0a0696f-6da2-4bf0-b482-af230fd3bc68", "target": "dorothy"}, {"pos": 7, "name": "Shut down workstation", "ability_id": "bddd018f-220e-4ce0-9c9a-17751199dd1a", "target": "dorothy"}, {"pos": 8, "name": "Delete file", "ability_id": "fed49260-7d04-4d9f-a918-b7dc6c4a98aa", "target": "dorothy"}, {"pos": 9, "name": "Delete registry entry", "ability_id": "96f40897-739b-4bbd-bef0-4adee91b6ee6", "target": "dorothy"}, {"pos": 10, "name": "Activate Firewall", "ability_id": "f01073d4-5992-49fa-9bd5-fa95bc7ab057", "target": "dorothy"}, {"pos": 11, "name": "Activate Windows Defender", "ability_id": "1dc816e7-525d-46d2-9b74-7dacc2da400e", "target": "dorothy"}, {"pos": 12, "name": "Block C2 communication", "ability_id": "0a512e0e-aa82-4419-8d23-3d550a769028", "target": "dorothy"}, {"pos": 13, "name": "Do nothing (leave system evolving)", "ability_id": "", "target": "toto"}, {"pos": 14, "name": "Disable user account", "ability_id": "ad0b71fe-1b2d-4847-a24a-f4de322ac360", "target": "toto"}, {"pos": 15, "name": "Force password reset", "ability_id": "4e5c5024-765d-4ff5-ae68-6d1d496c8bef", "target": "toto"}, {"pos": 16, "name": "Force sign out", "ability_id": "9fd2778f-91a9-469f-8357-0cb50b02a4ae", "target": "toto"}, {"pos": 17, "name": "Add to block list", "ability_id": "67741162-0d58-4021-bb60-0da5462db181", "target": "toto"}, {"pos": 18, "name": "Isolate endpoint", "ability_id": "040db47b-1a7b-439e-ae78-a80d5d5c5812", "target": "toto"}, {"pos": 19, "name": "Terminate process", "ability_id": "a0a0696f-6da2-4bf0-b482-af230fd3bc68",

# The SMALL one                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "target": "toto"}, {"pos": 20, "name": "Shut down workstation", "ability_id": "bddd018f-220e-4ce0-9c9a-17751199dd1a", "target": "toto"}, {"pos": 21, "name": "Delete file", "ability_id": "fed49260-7d04-4d9f-a918-b7dc6c4a98aa", "target": "toto"}, {"pos": 22, "name": "Delete registry entry", "ability_id": "96f40897-739b-4bbd-bef0-4adee91b6ee6", "target": "toto"}, {"pos": 23, "name": "Activate Firewall", "ability_id": "f01073d4-5992-49fa-9bd5-fa95bc7ab057", "target": "toto"}, {"pos": 24, "name": "Activate Windows Defender", "ability_id": "1dc816e7-525d-46d2-9b74-7dacc2da400e", "target": "toto"}, {"pos": 25, "name": "Block C2 communication", "ability_id": "0a512e0e-aa82-4419-8d23-3d550a769028", "target": "toto"}, {"pos": 26, "name": "Do nothing (leave system evolving)", "ability_id": "", "target": "wizard"}, {"pos": 27, "name": "Disable user account", "ability_id": "ad0b71fe-1b2d-4847-a24a-f4de322ac360", "target": "wizard"}, {"pos": 28, "name": "Force password reset", "ability_id": "4e5c5024-765d-4ff5-ae68-6d1d496c8bef", "target": "wizard"}, {"pos": 29, "name": "Force sign out", "ability_id": "9fd2778f-91a9-469f-8357-0cb50b02a4ae", "target": "wizard"}, {"pos": 30, "name": "Add to block list", "ability_id": "67741162-0d58-4021-bb60-0da5462db181", "target": "wizard"}, {"pos": 31, "name": "Isolate endpoint", "ability_id": "040db47b-1a7b-439e-ae78-a80d5d5c5812", "target": "wizard"}, {"pos": 32, "name": "Terminate process", "ability_id": "a0a0696f-6da2-4bf0-b482-af230fd3bc68", "target": "wizard"}, {"pos": 33, "name": "Shut down workstation", "ability_id": "bddd018f-220e-4ce0-9c9a-17751199dd1a", "target": "wizard"}, {"pos": 34, "name": "Delete file", "ability_id": "fed49260-7d04-4d9f-a918-b7dc6c4a98aa", "target": "wizard"}, {"pos": 35, "name": "Delete registry entry", "ability_id": "96f40897-739b-4bbd-bef0-4adee91b6ee6", "target": "wizard"}, {"pos": 36, "name": "Activate Firewall", "ability_id": "f01073d4-5992-49fa-9bd5-fa95bc7ab057", "target": "wizard"}, {"pos": 37, "name": "Activate Windows Defender", "ability_id": "1dc816e7-525d-46d2-9b74-7dacc2da400e", "target": "wizard"}, {"pos": 38, "name": "Block C2 communication", "ability_id": "0a512e0e-aa82-4419-8d23-3d550a769028", "target": "wizard"}]
ACTIONS = [
    {"pos": 0, "name": "Do nothing", "ability_id": "", "target": "dorothy"}, {"pos": 1, "name": "Isolate",
                                                                              "ability_id": "", "target": "dorothy"}, {"pos": 2, "name": "Delete file", "ability_id": "", "target": "dorothy"},
    {"pos": 3, "name": "Do nothing", "ability_id": "", "target": "toto"}, {"pos": 4, "name": "Isolate",
                                                                           "ability_id": "", "target": "toto"}, {"pos": 5, "name": "Delete file", "ability_id": "", "target": "toto"},
    {"pos": 6, "name": "Do nothing", "ability_id": "", "target": "wizard"}, {"pos": 7, "name": "Isolate",
                                                                             "ability_id": "", "target": "wizard"}, {"pos": 8, "name": "Delete file", "ability_id": "", "target": "wizard"}
]


class State:
    SOLUTIONS = ['Isolate', 'Shut down']

    def __init__(self, seed=None):
        self._lock = False
        self._obs = [0, 0, 0]
        self._attack = 0
        if seed is None:
            seed = round(random.random() * 20)
        for _ in range(seed):
            self.observe(0)
        # self._lock = True  # Enable to emulate STATIC fixed env

    def observe(self, action):
        # if (self._obs[1] == 2 and self._obs[2] == 1):
        #     print('here')
        if action >= len(ACTIONS):
            return self._obs[:]
        defense = ACTIONS[action]
        if action < len(ACTIONS) * (1 / 3):
            defender = 0
        elif action < len(ACTIONS) * (2 / 3):
            defender = 1
        else:
            defender = 2

        if self._attack < 10:
            attacker = 0
        elif self._attack < 20:
            attacker = 1
        else:
            attacker = 2

        solution = defense['name'] in State.SOLUTIONS
        self._attack += 1
        return self._update(attacker, defender, solution)[:]

    def _update(self, attacker, defender, solution):
        obs = self._obs
        increment = self._attack % 5 == 0 if not self._lock else False  # Increase damages from time to time
        if obs[defender] != 0 and solution:
            obs[defender] = 3
        if not increment:
            # print(obs)
            return obs[:]
        if obs[attacker] == 0:
            if attacker == 0:
                obs[attacker] += 1
            elif obs[attacker - 1] != 0 and obs[attacker - 1] != 3:
                obs[attacker] += 1
        elif obs[attacker] == 1:
            obs[attacker] += 1
        # print(obs)
        return obs[:]


# -------------------------- DUMMY ENV -------------------------

STATE = State(0)


def info():
    return {'actions': ACTIONS, 'observations': [0, 0, 0]}


def reset():
    global STATE
    STATE = State()
    return {'observation': STATE.observe(0), 'info': 'Reset environment'}


def step(action):
    return STATE.observe(action)


# ----------------------------- MAIN ----------------------------

if __name__ == "__main__":
    state = State(0)
    for r in range(40):
        print('>>> ', state.observe(random.randint(0, len(ACTIONS) - 1)))

    # state = State()
    # print('>>> ', state.observe(20))

    # print(info())
    # print(reset())
    # print(step(5))

    # STATE._obs = [3, 1, 0]
    # print(step(4))
