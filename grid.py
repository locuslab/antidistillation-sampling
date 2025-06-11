# -*- coding: utf-8 -*-
# grid.py
import random
import sys
import numpy as np

hostname = sys.argv[1]
hashmap = {'mezcal': 0, 'clairin': 1, 'baijiu': 2} # This is used to distribute the grid generation across multiple machines (identified by hostname)
offset = hashmap[hostname] if hostname in hashmap else 0 # If the hostname is not in the hashmap, we use 0 as the offset
lines = []
for tau in [0.9] + list(np.linspace(1.0, 2.0, 20)):
    if tau <= 1.0:
        for eps in [1e-2]:
            for b in list(np.linspace(0, 15, 20)):
                lam = b * eps
                lines += [f"{tau:.2e} {lam:.2e} {eps:.2e}\n"]
    else:
        eps = 0.0
        lam = 0.0
        lines += [f"{tau:.2e} {lam:.2e} {eps:.2e}\n"]

random.seed(42)
random.shuffle(lines)

print(''.join(lines[offset::len(hashmap)]))
