import datetime
import numpy as np
cimport numpy as np
from collections import defaultdict

#def get_accum(np.ndarray[long, ndim=3] phi, np.ndarray[long, ndim=3] psi, np.ndarray[double, ndim=3] edges, r_table):
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
