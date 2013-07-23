from __future__ import division
from numpy import log2
from collections import defaultdict, OrderedDict
from heapq import heappush, heappop, heapify

from prob import FiniteRandomVariable

################
#  iid coding  #
################

class IIDCode(object):
    def __init__(self,codebook):
        self._codebook = codebook

    def compress(self,seq):
        for symbol in seq:
            yield self._codebook[symbol]

    def __repr__(self):
        return '\n'.join('%s -> %s' % (symbol, code)
                for symbol, code in self._codebook.iteritems())

    @classmethod
    def from_rv(cls,X):
        return cls(huffman(X))

    @staticmethod
    def fit_and_compress(seq,blocklen=1):
        seq = blockify(seq,blocklen)
        model = estimate_iid_source(seq)
        code = IIDCode.from_rv(model)
        result = ''.join(code.compress(seq))

        print code

        N = len(list(seq))
        inbits = N*log2(len(model.support()))
        outbits = len(result)
        print '\ncompression rate: %gx (%g bits per symbol with blocklength of %d)\n' \
                % (outbits/inbits, outbits/N/blocklen, blocklen)

        return result


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

def estimate_iid_source(seq):
    counts = defaultdict(int)
    tot = 0
    for symbol in seq:
        counts[symbol] += 1
        tot += 1
    pmf = {symbol:count/tot for symbol, count in counts.iteritems()}
    return FiniteRandomVariable(pmf)

def blockify(seq,blocklen):
    return [seq[i:i+blocklen] for i in xrange(len(seq)//blocklen)]

