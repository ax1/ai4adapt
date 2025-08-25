from gymnasium import spaces
import numpy as np

box = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
print(box.sample())
# [0.03029283 0.22555362 0.02997451 0.8011286]

discrete = spaces.Discrete(2)  # discrete must be 0..n Natural numbers
print(discrete.contains(0))
for i in range(100):
    print(discrete.sample())
