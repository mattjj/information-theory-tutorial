from __future__ import division
import numpy as np
from numpy import log2
from numpy.random import choice as sample
import abc
from collections import deque
from itertools import islice

import util

######################
#  random variables  #
######################

# a finite random variable is specified by a pmf

class RV(object):
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

class BinaryRV(RV):
    def __init__(self,p):
        super(BinaryRV,self).__init__({0:1-p,1:p})

######################################################
#  entropy and relative entropy of random variables  #
######################################################

def H(X):
    p = X.pmf
    return sum(-p(x) * log2(p(x)) for x in X.range() if p(x) > 0)

def KL(X,Y):
    p,q = X.pmf, Y.pmf
    return sum(p(x) * (log2(p(x)) - log2(q(x))) for x in X.range() if p(x) > 0)

###############
#  processes  #
###############

class Process(object):
    __metaclass__ = abc.ABCMeta

    def sample_sequence(self,N):
        return ''.join(islice(self.sequence_generator(),N))

    @abc.abstractmethod
    def sequence_generator(self):
        pass

    @abc.abstractmethod
    def H_rate(self):
        pass

def H_rate(process):
    return process.H_rate()

# an iid process is just a wrapped random variable

class IIDProcess(Process):
    def __init__(self,pmf):
        self._rv = RV(pmf)

    def sequence_generator(self):
        X = self._rv
        while True:
            yield sample(X.range(),p=X._pmf.values())

    def H_rate(self):
        return H(self._rv)

    # for efficiency
    def sample_sequence(self,N):
        X = self._rv
        return sample(X.range(),N,p=X._pmf.values())

class MarkovProcess(Process):
    def __init__(self,symbols,trans):
        self._symbols = symbols
        self._numstates = len(symbols)
        self._P = np.asarray(trans)
        self._pi = util.steady_state(self._P)

    def sequence_generator(self):
        state = sample(self._numstates,p=self._pi)
        while True:
            yield self._symbols[state]
            state = sample(self._numstates,p=self._P[state])

    def H_rate(self):
        P = self._P
        PlogP = P*log2(np.where(P != 0, P, 1))
        return -self._pi.dot(PlogP).sum()

