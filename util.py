from __future__ import division
import numpy as np

def steady_state(P):
    evals, evecs = np.linalg.eig(P.T)
    pi = evecs[:,np.abs(evals).argmax()]
    pi /= pi.sum()
    return pi


