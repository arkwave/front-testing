from classes import *
f1 = Future('may', 'SX17', 17)
f2 = Future('may', 'SX15', 16)
f3 = Future('may', 'SX14', 15)
f4 = Future('may', 'SX13', 14)

s1 = Option(10, 50, 'call', 10, f1, 'euro')
s2 = Option(10, 40, 'call', 10, f2, 'euro')
s3 = Option(10, 30, 'call', 10, f3, 'euro')
s4 = Option(10, 10, 'call', 10, f3, 'euro')
s5 = Option(10, 3, 'call', 10, f4, 'euro')
optionList = [s1, s2]


# testing basic functionality for securities.
print(s1.compute_value())
print(s1.greeks())


# testing basic functionality.
pf = Portfolio(optionList, [])
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

optionList = [s1, s2, s3, s4, s5]
pf2 = Portfolio(optionList, [])
print(pf2.get_underlying())
print(pf2.get_securities_monthly())


# testing exercising.
s10 = Option(10, 50, 'call', 10, f1, 'euro')
pf3 = Portfolio([s10], [])
print(pf3.get_securities_monthly())
pf3.exercise_option(s10)
print(pf3.get_securities_monthly())

# testing passage of time.
s10 = Option(10, 50, 'call', 10, f1, 'euro')
pf3 = Portfolio([s10], [])
print(s10.tau)
pf3.timestep(1)
print(s10.tau)
