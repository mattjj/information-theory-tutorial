```python
In [1]: run test

### IID Process ###

true entropy rate: 1.57095

a -> 1
b -> 00
c -> 010
d -> 011

IIDCode with block length 1 achieved compression rate: 0.80325x
    (2 bits per raw symbol, 1.6065 compressed bits per symbol)


### Markov Process ###

true entropy rate: 0.695462

b -> 00
c -> 01
d -> 10
a -> 11

IIDCode with block length 1 achieved compression rate: 1x
    (2 bits per raw symbol, 2 compressed bits per symbol)

cc -> 00
dd -> 01
aa -> 10
bb -> 111
ba -> 11000
ab -> 11001
cd -> 110100
dc -> 110101
bc -> 110110
cb -> 110111

IIDCode with block length 2 achieved compression rate: 0.6761x
    (4 bits per raw symbol, 2.7044 compressed bits per symbol)
```
