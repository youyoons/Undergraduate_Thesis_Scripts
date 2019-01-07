import test
import numpy as np
import datetime
from collections import defaultdict

phi = np.random.randint(180, size = (80,80,40))
psi = np.random.randint(180, size = (80,80,40))
edges = np.ones((80,80,40))
r_table = defaultdict(list)

for i in range(180):
    for j in range(180):
        r_table[i,j].append((i,j,i+j))
        r_table[i,j].append((i,j,i+2*j))
        #r_table[i,j].append((i,2*j,i+j))
        #r_table[i,j].append((2*i,j+10,i+j+3))

accumulator = test.get_accum(phi,psi,edges,r_table)
