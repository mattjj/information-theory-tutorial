from __future__ import division
import numpy as np
from numpy import log2
from numpy.random import choice
import abc
from collections import deque

import util

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

    def range(self):
        return self._pmf.keys()

    def sample(self,N=1):
        return choice(self.range(),N,p=self._pmf.values())

######################################################
#  entropy and relative entropy of random variables  #
######################################################

def H(X):
    p = X.pmf
    return sum(-p(x) * log2(p(x)) for x in X.range())

def D(X,Y):
    p,q = X.pmf, Y.pmf
    return sum(p(x) * (log2(p(x)) - log2(q(x))) for x in X.range())

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
        self._pi = util.steady_state(self._P)

    def generate_sequence(self,N):
        out = deque()
        state = 0 # by convention, always start in 0
        for n in xrange(N):
            out.append(self._symbols[state])
            state = choice(self._numstates,p=self._P[state])
        return ''.join(str(x) for x in out)

    def H_rate(self):
        P = np.where(self._P != 0, self._P, 1)
        errs = np.seterr(divide='ignore')
        out = -self._pi.dot(self._P * log2(P)).sum()
        np.seterr(**errs)
        return out

