import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from types import SimpleNamespace

par = SimpleNamespace()
par.J = 3
par.N = 10
par.K = 10000

par.F = np.arange(1,par.N+1)
par.sigma = 2

par.v = np.array([1,2,3])
par.c = 1


def epsilon(n=par.K):
    n = n # Amount of simulations
    eps = np.random.normal(loc=0, scale = par.sigma, size = n) # normal distribution
    return eps

def career_simulation(i):
    '''Career choice simulation for person i '''
    results = [] # Empty list to store results
    i = i # i is parameter person with i friends
    
    for _ in range(par.K): # loop 10.000 times

        # Draw friend errors
        e_j1 = epsilon(i)
        e_j2 = epsilon(i)
        e_j3 = epsilon(i)

        # Calculating observed priors for friends
        u1_j = 1 + np.mean(e_j1) 
        u2_j = 2 + np.mean(e_j2)
        u3_j = 3 + np.mean(e_j3)

        # Indexing draws
        u_j = [u1_j, u2_j, u3_j]
        j = [1, 2, 3]

        # Choice of career j
        u_max_index = np.argmax(u_j) # Index with maximum prior
        j_max = j[u_max_index] # Optimal career j choice with maximum prior of friends
        u_prior = u_j[u_max_index] # Prior value for career j

        # Draw own errors
        e_i1 = epsilon(1)
        e_i2 = epsilon(1)
        e_i3 = epsilon(1)

        # Realized career utilities
        u1_i = 1 + e_i1
        u2_i = 2 + e_i2
        u3_i = 3 + e_i3

        u_i = [u1_i, u2_i, u3_i] # Listing career
        realized_j = u_i[j_max-1] # Own realized career utility
        
        results.append((j_max, u_prior, realized_j))

    return results

def career_simulation_new(i):
    '''Career choice simulation with learning and switching'''
    results = []
    i = i
    
    for _ in range(par.K):

        # Draw friend errors
        e_j1 = epsilon(i)
        e_j2 = epsilon(i)
        e_j3 = epsilon(i)

        # Calculating priors
        u1_j = 1 + np.mean(e_j1)
        u2_j = 2 + np.mean(e_j2)
        u3_j = 3 + np.mean(e_j3)

        # Indexing draws
        u_j = [u1_j, u2_j, u3_j]
        j = [1, 2, 3]

        # Indexing to choose correct J career move
        u_max_index = np.argmax(u_j)
        j_max = j[u_max_index]
        u_prior = u_j[u_max_index]

        # Draw own errors
        e_i1 = epsilon(1)
        e_i2 = epsilon(1)
        e_i3 = epsilon(1)

        # Realized careers
        u1_i = 1 + e_i1
        u2_i = 2 + e_i2
        u3_i = 3 + e_i3

        u_i = [u1_i, u2_i, u3_i]
        realized_j = u_i[j_max-1]

        # ---------------
        # Learning and switching

        # New utilities
        if j_max == 1:
            u1_new = u1_i[0] # If staying then expected utility is equal to realized
        else:
            u1_new = u1_j - par.c # If switching to then expected equal to prior - 1 

        if j_max == 2:
            u2_new = u2_i[0]
        else:
            u2_new = u2_j - par.c

        if j_max == 3:
            u3_new = u3_i[0]
        else:
            u3_new = u3_j - par.c

        # Indexing draws
        u_new = [u1_new, u2_new, u3_new]

        # Indexing to choose correct j career move
        u_max_index_new = np.argmax(u_new)
        j_max_new = j[u_max_index_new]
        u_prior_new = u_new[u_max_index_new]

        # Realized new utility
        if j_max_new == j_max:
            realized_j_new = u_i[j_max_new-1]
        else:
            realized_j_new = u_i[j_max_new-1] - 1


        results.append((j_max, j_max_new, u_prior, u_prior_new, realized_j, realized_j_new))

    return results