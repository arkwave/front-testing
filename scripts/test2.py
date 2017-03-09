'''
TODO: Implement simulation procedure.


Things to keep in mind:

1) Daily vs Bullet. 
2) Barriers require different data and handling.
3) American vs European barriers.
4) Expiry dates. Decrement tau appropriately for all options.
5) Deriving the options to buy to hedge gamma/theta.
6) Multiple runs. Stick everything in a for loop. Final output = histogram of PnLs acquired.
	> where is the stochasticity? from the price/vol series?

'''

gamma_max =
gamma_min =
theta_max =
theta_min =

from classes import *

# Set up the portfolio on which to run simulation.
s1 = Option()
s2 = Option()
s3 = Option()
s4 = Option()
s5 = Option()
optionList = [s1, s2, s3, s4, s5]
pf = Portfolio(optionList)

# run simulation on the portfolio
numIterations = 1000
for i in range(numIterations):
    # do stuff.
    pass
