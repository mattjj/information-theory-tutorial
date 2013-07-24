from __future__ import division
import numpy as np
from numpy import log2
import abc
from collections import defaultdict, OrderedDict
from heapq import heappush, heappop, heapify
from itertools import izip, tee

from prob import FiniteRandomVariable

##########
#  util  #
##########

def blockify(seq,blocklen):
    return [seq[i:i+blocklen] for i in xrange(len(seq)//blocklen)]

def huffman(X):
    p = X.pmf

    queue = [(p(x),(x,),('',)) for x in X.support()]
    heapify(queue)

    while len(queue) > 1:
        p2, symbols1, codes1 = heappop(queue)
        p1, symbols2, codes2 = heappop(queue)
        heappush(queue, (p1+p2, symbols1+symbols2,
            ['0' + c for c in codes1] + ['1' + c for c in codes2]))

    _, symbols, codes = queue[0]
    return OrderedDict(sorted(zip(symbols, codes),key=lambda x: len(x[1])))

###########
#  codes  #
###########

# a code specifies a mapping from symbols to codewords

class Code(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compress(self,seq):
        pass

    @abc.abstractmethod
    def decompress(self,bitstring):
        pass

# a model-based code corresponds to fitting a probabilistic model to data

class ModelBasedCode(Code):
    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractmethod
    def fit(cls,seq,blocklen=1):
        pass

    @classmethod
    def fit_and_compress(cls,seq,blocklen=1):
        seq = blockify(seq,blocklen)
        code = cls.fit(seq)
        result = ''.join(code.compress(seq))

        print code

        N = len(seq)
        bits_per_input_symbol = log2(len(set(seq)))
        inbits = N*bits_per_input_symbol
        outbits = len(result)
        print '\n%s compression rate: %gx\n    (with blocklength %d, 2^%g symbols, %g bits per symbol)\n' \
                % (cls.__name__, outbits/inbits,
                        blocklen, bits_per_input_symbol, outbits/N/blocklen)

        return result

# an iid code is backed by an iid random variable model and a codebook

class IIDCode(ModelBasedCode):
    def __init__(self,codebook):
        self._codebook = codebook
        self._inv_codebook = {v:k for k,v in codebook.iteritems()}

    def compress(self,seq):
        return ''.join(self._codebook[s] for s in seq)

    def decompress(self,bits):
        bits = iter(bits)
        while True:
            yield self.consume_next(self._inv_codebook,bits)

    @staticmethod
    def consume_next(inv_codebook,bits):
        # NOTE: prefix-free is important here!
        bitbuf = ''
        for bit in bits:
            bitbuf += bit
            if bitbuf in inv_codebook:
                return inv_codebook[bitbuf]

    def __repr__(self):
        return '\n'.join('%s -> %s' % (symbol, code)
                for symbol, code in self._codebook.iteritems())

    @classmethod
    def fit(cls,seq):
        model = cls.estimate_iid_source(seq)
        return cls.from_rv(model)

    @classmethod
    def from_rv(cls,X):
        return cls(huffman(X))

    @staticmethod
    def estimate_iid_source(seq):
        counts = defaultdict(int)
        tot = 0
        for symbol in seq:
            counts[symbol] += 1
            tot += 1
        pmf = {symbol:count/tot for symbol, count in counts.iteritems()}
        return FiniteRandomVariable(pmf)

# a Markov code is backed by a Markov chain model and for each symbol it has a
# separate codebook to encode the next symbol
# TODO test all this
# TODO pass in alphabet to estimation methods

class MarkovCode(object):
    def __init__(self,codebooks):
        self._codebooks = codebooks # dict of codebooks, indexed by alphabet
        self._firstcodebook = codebooks.items()[0] # convention
        self._inv_firstcodebook = {v:k for k,v in self._firstcodebook.iteritems()}
        self._inv_codebooks = {v:{vv:kk for kk,vv in v.iteritems()}
                for k,v in codebooks.iteritems()}

    def compress(self,seq):
        seq1, seq2 = tee(seq,2)
        firstsymbol = next(seq2)
        return self._firstcodebook[firstsymbol] + \
                ''.join(self._codebooks[s1][s2] for s1, s2 in izip(seq1,seq2))

    def decompress(self,bits):
        bits = iter(bits)
        symbol = IIDCode.consume_next(self._firstcodebook,bits)
        while True:
            yield symbol
            symbol = IIDCode.consume_next(self._inv_codebooks[symbol],bits)

    @classmethod
    def fit(cls,seq):
        model = cls.estimate_markov_source(seq)
        return cls.from_markovchain(model)

    @classmethod
    def from_markovchain(cls,(symbols,P)):
        return cls({s:huffman(FiniteRandomVariable(dict(*zip(symbols,dist))))
            for s,dist in zip(symbols,P)})

    @staticmethod
    def estimate_markov_source(seq):
        counts = defaultdict(lambda: defaultdict(int))
        tots = defaultdict(int)
        s1, s2 = tee(seq,2)
        _ = next(s2)
        for a,b in izip(s1,s2):
            counts[s1][s2] += 1
            tots[s1] += 1
        symbols = counts.keys()
        return (symbols, np.array([[counts[i][j]/tots[i] for j in symbols] for i in symbols]))

