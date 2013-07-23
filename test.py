from prob import *
from compress import *

print '\n### IID Process ###\n'
process = IIDProcess({'a':0.6,'b':0.2,'c':0.1,'d':0.1})
print '\ntrue entropy rate: %g\n' % H_rate(process)
seq = process.generate_sequence(10000)
IIDCode.fit_and_compress(seq)

print '\n### Markov Process ###\n'
process = MarkovProcess(('a','b','c','d'),np.array([[0.9,0.1,  0,  0],
                                                    [0.1,0.8,0.1,  0],
                                                    [  0,0.1,0.8,0.1],
                                                    [  0,  0,0.1,0.9]]))
print '\ntrue entropy rate: %g\n' % H_rate(process)
seq = process.generate_sequence(10000)
IIDCode.fit_and_compress(seq,blocklen=2)
