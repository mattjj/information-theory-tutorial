from __future__ import division
import numpy as np
from numpy import log2, ceil
import abc
from collections import defaultdict, OrderedDict
from heapq import heappush, heappop, heapify
from itertools import izip, izip_longest, imap, tee

import util
from prob import RV

##########
#  util  #
##########

def blockify(seq,blocklen,fill=None):
    args = [iter(seq)]*blocklen
    return imap(''.join, izip_longest(*args,fillvalue=fill))

def huffman(X):
    p = X.pmf

    # make a queue of symbols that we'll group together as we build the tree
    queue = [(p(x),(x,)) for x in X.range()]
    heapify(queue)
    # the initial codes are all empty
    codes = {x:'' for x in X.range()}

    while len(queue) > 1:
        # take the two lowest probability (grouped) symbols
        p1, symbols1 = heappop(queue)
        p2, symbols2 = heappop(queue)

        # add a bit to each symbol group's codes
        for s in symbols1:
            codes[s] = '0' + codes[s]
        for s in symbols2:
            codes[s] = '1' + codes[s]

        # merge the two groups and push them back to the queue
        heappush(queue, (p1+p2, symbols1+symbols2))

    # return them ordered by code length
    return OrderedDict(sorted(codes.items(),key=lambda x: len(x[1])))

###########
#  codes  #
###########

class Code(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compress(self,seq):
        pass

    @abc.abstractmethod
    def decompress(self,bitstring):
        pass

#######################
#  model-based codes  #
#######################

# a model-based code corresponds to fitting a probabilistic model to data
# (this class combines the internals of the model with the coding mechanism
# based on huffman codes, but arithmetic coding will provide the interfaces
# needed to separate them)

class ModelBasedCode(Code):
    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractmethod
    def fit(cls,seq,blocklen=1):
        pass

    # the next method is for convenience TODO move the printing parts elsewhere

    @classmethod
    def fit_and_compress(cls,seq,blocklen=1):
        if blocklen != 1:
            seq = list(blockify(seq,blocklen))

        code = cls.fit(seq)
        compressed = ''.join(code.compress(seq))
        decompressed = ''.join(code.decompress(compressed))

        assert decompressed == ''.join(seq)

        print code

        N = len(seq)
        bits_per_input_symbol = ceil(log2(len(set(seq))))
        inbits = N*bits_per_input_symbol
        outbits = len(compressed)
        print '\n%s with block length %d achieved compression rate: %gx\n    (%g bits per raw symbol, %g compressed bits per symbol)\n' \
                % (cls.__name__, blocklen, outbits/inbits,
                        bits_per_input_symbol, outbits/N)

        return compressed

# an iid code is backed by an iid random variable model and a huffman-derived
# codebook

class IIDCode(ModelBasedCode):
    def __init__(self,codebook):
        self.codebook = codebook
        self.inv_codebook = {v:k for k,v in codebook.iteritems()}

    def compress(self,seq):
        for s in seq:
            yield self.codebook[s]

    def decompress(self,bits):
        bits = iter(bits)
        while True:
            yield self.consume_next(self.inv_codebook,bits)

    @staticmethod
    def consume_next(inv_codebook,bits):
        # NOTE: prefix-free is important here!
        bitbuf = ''
        for bit in bits:
            bitbuf += bit
            if bitbuf in inv_codebook:
                return inv_codebook[bitbuf]
        assert len(bitbuf) == 0
        raise StopIteration

    def __repr__(self):
        return self.__class__.__name__ + '\n' + \
                '\n'.join('%s -> %s' % (symbol, code)
                        for symbol, code in self.codebook.iteritems())

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
        return RV(pmf)

# a Markov code is backed by a Markov chain model and for each symbol it has a
# separate huffman-derived codebook to encode the next symbol

class MarkovCode(ModelBasedCode):
    def __init__(self,firstcodebook,codebooks):
        self.firstcode = IIDCode(firstcodebook)
        self.iid_codes = {symbol:IIDCode(codebook)
                for symbol, codebook in codebooks.iteritems()}

    def compress(self,seq):
        s1, s2 = tee(seq,2)
        firstsymbol = next(s2)
        yield self.firstcode.codebook[firstsymbol]
        for a,b in izip(s1,s2):
            yield self.iid_codes[a].codebook[b] 

    def decompress(self,bits):
        bits = iter(bits)
        symbol = IIDCode.consume_next(self.firstcode.inv_codebook,bits)
        while True:
            yield symbol
            symbol = IIDCode.consume_next(self.iid_codes[symbol].inv_codebook,bits)

    def __repr__(self):
        return self.__class__.__name__ + '\n' + \
                '\n'.join('Code after seeing %s:\n%s' % (symbol,iidcode)
                        for symbol, iidcode in self.iid_codes.iteritems())

    @classmethod
    def fit(cls,seq):
        model = cls.estimate_markov_source(seq)
        return cls.from_markovchain(model)

    @classmethod
    def from_markovchain(cls,(symbols,pi,P)):
        return cls(huffman(RV(dict(zip(symbols,pi)))),
                {s:huffman(RV(dict(zip(symbols,dist))))
            for s,dist in zip(symbols,P)})

    @staticmethod
    def estimate_markov_source(seq):
        s1, s2 = tee(seq,2)
        next(s2)
        counts = defaultdict(lambda: defaultdict(int))
        tots = defaultdict(int)
        for a,b in izip(s1,s2):
            counts[a][b] += 1
            tots[a] += 1
        symbols = counts.keys()
        P = np.array([[counts[i][j]/tots[i] for j in symbols] for i in symbols])
        pi = util.steady_state(P)
        return (symbols, pi, P)

# it would be preferable to separate the probabilistic model, which does the
# fitting and predicting, from the coding mechanism. but huffman coding only
# gives us a codebook for a fixed prediction distribution, so we'd have to run
# the huffman algorithm for each possible prediction distribution our model can
# produce. arithmetic coding provides the tweak on the entropy coding / Kraft
# tree idea that enables such a separation of the probabilistic model and the
# code.

class ProbabilisticPredictionModel(object):
    pass # TODO

class ArithmeticCode(Code):
    pass # TODO, maybe mixin to generate classes that inherit from ModelBasedCode ?

#######################
#  model-free coding  #
#######################

# many of the best coding schemes don't fit probabilistic models at all, at
# least not explicitly. instead, based on ideas by Lempel and Ziv in 1977-1978,
# they produce a codebook based on substrings in the sequence itself. this
# approach is asymptotically optimal for stationary sources, so it's kind of
# like implicitly fitting a really big HMM, but it short-circuits the actual
# fitting step so that one doesn't need to be constrained by model choices (e.g.
# state sizes, Markov order) or fitting algorithms (instead, the constraint is
# usually in terms of window/dict size). without fitting an explicit
# probabilistic model, we have no model to inspect or make predictions with, but
# these codes work really well

class LempelZiv(Code):
    def __init__(self,alphabet):
        self._initial_dict = {symbol:bin(i)[2:] for i,symbol in enumerate(alphabet)}
        self._initial_invdict = {bits:symbol for symbol, bits in self._initial_dict.iteritems()}

    def compress(self,seq):
        codebook = self._initial_dict.copy()

        current_string = '' # symbol string
        for s in seq:
            new_string = current_string + s
            if new_string in codebook:
                # we've seen the new string before! we can keep growing
                current_string = new_string
            else:
                # we haven't seen the new string before
                yield codebook[current_string] # yield the part we had seen
                codebook[new_string] = bin(len(codebook))[2:] # new encoding for new string
                current_string = s # start building a new string

        # return any remaining amount we've buffered
        if len(current_string) > 0:
            yield codebook[current_string]

    def decompress(self,seq):
        inv_codebook = self._initial_invdict.copy()

        current_string = '' # bit string
        for s in seq:
            # the compressor is feeding us a bit code for something in our
            # dictionary. since the dictionary is ordered, we can keep consuming
            # until we see something not in our dictionary. then we've gone too
            # far, and we need to emit the part we can handle, then learn
            # something new: that the next (numerically) dict entry is the
            # symbol string we just emitted plus the first symbol of the NEXT
            # symbol string we decode.

            pass # TODO
