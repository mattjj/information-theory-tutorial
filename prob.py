from __future__ import division
import numpy as np
from numpy import log2
from numpy.random import choice
from numpy.linalg import eig
import abc
from collections import deque

np.seterr(divide='ignore')

######################
#  random variables  #
######################

# a finite random variable is specified by a pmf

class FiniteRandomVariable(object):
    def __init__(self,pmf):
        self._pmf = pmf

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__,self._pmf)

    def pmf(self,x):
        if x in self._pmf:
            return self._pmf[x]
        else:
            return 0

    def support(self):
        return self._pmf.keys()

    def sample(self,N=1):
        return choice(self.support(),N,p=self._pmf.values())

######################################################
#  entropy and relative entropy of random variables  #
######################################################

def H(X):
    p = X.pmf
    return sum(-p(x) * log2(p(x)) for x in X.support())

def D(X,Y):
    p,q = X.pmf, Y.pmf
    return sum(p(x) * (log2(p(x)) - log2(q(x))) for x in X.support())

###############
#  processes  #
###############

class Process(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_sequence(self,N):
        pass

    @abc.abstractmethod
    def H_rate(self):
        pass

def H_rate(process):
    return process.H_rate()

# an iid process is just a wrapped random variable

class IIDProcess(object):
    def __init__(self,pmf):
        self._rv = FiniteRandomVariable(pmf)

    def generate_sequence(self,N):
        X = self._rv
        return ''.join(str(x) for x in X.sample(N))

    def H_rate(self):
        return H(self._rv)

class MarkovProcess(object):
    def __init__(self,symbols,trans):
        self._symbols = symbols
        self._numstates = len(symbols)
        self._P = np.asarray(trans)

        evals, evecs = eig(self._P.T)
        self._pi = evecs[:,np.abs(evals).argmax()]
        self._pi /= self._pi.sum()

    def generate_sequence(self,N):
        out = deque()
        state = 0 # by convention, always start in 0
        for n in xrange(N):
            out.append(self._symbols[state])
            state = choice(self._numstates,p=self._P[state])
        return ''.join(str(x) for x in out)

    def H_rate(self):
        P = np.where(self._P != 0, self._P, 1)
        return -self._pi.dot(self._P * log2(P)).sum()

