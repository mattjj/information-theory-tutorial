```python
In [1]: run test

### IID Process ###

true entropy rate: 1.57095

IIDCode
a -> 1
b -> 00
c -> 010
d -> 011

IIDCode with block length 1 achieved compression rate: 0.805x
    (2 bits per raw symbol, 1.61 compressed bits per symbol)


### Markov Process ###

true entropy rate: 0.695462

IIDCode
a -> 00
b -> 01
c -> 10
d -> 11

IIDCode with block length 1 achieved compression rate: 1x
    (2 bits per raw symbol, 2 compressed bits per symbol)

IIDCode
aa -> 00
cc -> 01
dd -> 10
bb -> 111
cb -> 11000
dc -> 11001
bc -> 110100
ba -> 110101
ab -> 110110
cd -> 110111

IIDCode with block length 2 achieved compression rate: 0.68095x
    (4 bits per raw symbol, 2.7238 compressed bits per symbol)

MarkovCode
Code after seeing a:
IIDCode
a -> 1
b -> 01
c -> 000
d -> 001
Code after seeing c:
IIDCode
c -> 1
d -> 01
a -> 000
b -> 001
Code after seeing b:
IIDCode
b -> 1
a -> 01
d -> 000
c -> 001
Code after seeing d:
IIDCode
d -> 1
c -> 01
a -> 000
b -> 001

MarkovCode with block length 1 achieved compression rate: 0.595025x
    (2 bits per raw symbol, 1.19005 compressed bits per symbol)
```
