import numpy as np
from numba import jit
from datetime import datetime

@jit
def update(spins, random, random_step, E, T):
    L = spins.shape[0]
    for r in range(L):
        for c in range(L):
            s = spins[r,c]
            e0 = -(np.cos(s-spins[(r-1)%L,c])+np.cos(s-spins[(r+1)%L,c])+
                   np.cos(s-spins[r,(c-1)%L])+np.cos(s-spins[r,(c+1)%L]))
            t = s + 2*(random_step[r,c]-0.5)*np.pi*1.
            e1 = -(np.cos(t-spins[(r-1)%L,c])+np.cos(t-spins[(r+1)%L,c])+
                   np.cos(t-spins[r,(c-1)%L])+np.cos(t-spins[r,(c+1)%L]))
            dE = e1 - e0
            E[r,c] = e0
            if np.exp(-dE/T) > random[r,c]:
                spins[r,c] = t
                E[r,c] = e1
    return spins, E
                
def runmc(spins, T, itns):
    L = spins.shape[0]
    E = np.zeros_like(spins)
    start = datetime.now()
    for i in xrange(itns):
        update(spins, np.random.rand(L,L), np.random.rand(L,L), E, T)
    print(datetime.now() - start)
    return spins, E

def chessupdate(spins, random, random_step, dE_m, T):
    L = spins.shape[0]
    idx = np.arange(L*L)
    chess = np.array([((i%L)%2 + (i/L)%2)%2 for i in idx])
    chessidx = np.r_[idx[chess==0], idx[chess==1]]
    print(chessidx)
    T_M = np.zeros_like(spins)
    for i in chessidx:
        r, c = i%L, i/L
        s = spins[r,c]
        e0 = -(np.cos(s-spins[(r-1)%L,c])+np.cos(s-spins[(r+1)%L,c])+
               np.cos(s-spins[r,(c-1)%L])+np.cos(s-spins[r,(c+1)%L]))
        t = s + 2*(random_step[r,c]-0.5)*np.pi*1.
        T_M[r,c] = t
        e1 = -(np.cos(t-spins[(r-1)%L,c])+np.cos(t-spins[(r+1)%L,c])+
               np.cos(t-spins[r,(c-1)%L])+np.cos(t-spins[r,(c+1)%L]))
        dE = e1 - e0
        dE_m[r,c] = dE
        if np.exp(-dE/T) > random[r,c]:
            spins[r,c] = t
    return spins, dE_m, T_M
 
