from __future__ import division
from numpy import log2
import abc
from collections import defaultdict, OrderedDict
from heapq import heappush, heappop, heapify

from prob import FiniteRandomVariable

##########
#  util  #
##########

def blockify(seq,blocklen):
    return [seq[i:i+blocklen] for i in xrange(len(seq)//blocklen)]

###########
#  codes  #
###########

# a code specifies a mapping from symbols to codewords

class Code(object):
    def __init__(self,codebook):
        self._codebook = codebook

    def compress(self,seq):
        for symbol in seq:
            yield self._codebook[symbol]

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

        N = len(list(seq))
        bits_per_input_symbol = log2(len(set(seq)))
        inbits = N*bits_per_input_symbol
        outbits = len(result)
        print '\n%s compression rate: %gx\n    (with blocklength %d, 2^%g symbols, %g bits per symbol)\n' \
                % (cls.__name__, outbits/inbits,
                        blocklen, bits_per_input_symbol, outbits/N/blocklen)

        return result


class IIDCode(ModelBasedCode):
    def __repr__(self):
        return '\n'.join('%s -> %s' % (symbol, code)
                for symbol, code in self._codebook.iteritems())

    @classmethod
    def fit(cls,seq,blocklen=1):
        model = cls.estimate_iid_source(seq)
        return cls.from_rv(model)

    @classmethod
    def from_rv(cls,X):
        return cls(IIDCode.huffman(X))

    @staticmethod
    def estimate_iid_source(seq):
        counts = defaultdict(int)
        tot = 0
        for symbol in seq:
            counts[symbol] += 1
            tot += 1
        pmf = {symbol:count/tot for symbol, count in counts.iteritems()}
        return FiniteRandomVariable(pmf)

    @staticmethod
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


class MarkovCode(object):
    pass

