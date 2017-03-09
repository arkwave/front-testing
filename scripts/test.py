from classes import *

s1 = Option(10, 20, 50, 'call', 10, 17, 'may')
s2 = Option(10, 5, 40, 'call', 10, 16, 'may')
s3 = Option(10, 15, 30, 'call', 10, 18, 'may')
s4 = Option(10, 10, 10, 'call', 10, 19, 'may')
s5 = Option(10, 8, 3, 'call', 10, 20, 'may')
optionList = [s1, s2]


# testing basic functionality for securities.
print(s1.compute_value())
print(s1.greeks())


# testing basic functionality.
pf = Portfolio(optionList)
val = pf.compute_value()
print(val)
print(val == pf.compute_value())
print(pf.get_securities_monthly())
print(pf.get_securities())


pf.add_security(s2)  # should be 3 securities. works!
print(len(pf.get_securities()))

pf.add_security(s3)
val2 = pf.compute_value()
print(val2 == val)
print(val2)
print(pf.get_securities_monthly())
print(pf.get_securities())

pf.remove_security(s3)
pf.remove_security(s2)
val3 = pf.compute_value()
print(val3 == val)
print(pf.get_securities_monthly())
print(pf.get_securities())
