import datetime
import numpy as np
from collections import defaultdict

def get_accum(phi, psi, edges, r_table):
    t1 = datetime.datetime.now()
    accum_i = 0
    accum_j = 0
    accum_k = 0
    edges_dim = np.shape(edges)
    accumulator = np.zeros(edges_dim)


    for (i,j,k),value in np.ndenumerate(edges):
        if value: 
            for r in r_table[(int(phi[i,j,k]), int(psi[i,j,k]))]:
                accum_i, accum_j, accum_k = i+r[0], j+r[1], k+r[2]
                        
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1] and accum_k < accumulator.shape[2]:
                    accumulator[int(accum_i), int(accum_j), int(accum_k)] += 1 

    t2 = datetime.datetime.now()
    print(t2-t1)
    
    return accumulator



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

accumulator = get_accum(phi,psi,edges,r_table)
