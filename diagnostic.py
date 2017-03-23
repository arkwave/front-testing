char = 'call'
strike = 300
tau = 0.89589041
vol = 0.19142796
s = 380
r = 0
product = 'C'
payoff = 'euro'
ki = None
ko = None
barrier = None
direction = None

from math import log, exp, pi, sqrt
from scipy.stats import norm

multipliers = {

    'LH':  [22.046, 18.143881, 0.025, 0.05, 400],
    'LSU': [1, 50, 0.1, 10, 50],
    'LCC': [1.2153, 10, 1, 25, 12.153],
    'SB':  [22.046, 50.802867, 0.01, 0.25, 1120],
    'CC':  [1, 10, 1, 50, 10],
    'CT':  [22.046, 22.679851, 0.01, 1, 500],
    'KC':  [22.046, 17.009888, 0.05, 2.5, 375],
    'W':   [0.3674333, 136.07911, 0.25, 10, 50],
    'S':   [0.3674333, 136.07911, 0.25, 10, 50],
    'C':   [0.3936786, 127.00717, 0.25, 10, 50],
    'BO':  [22.046, 27.215821, 0.01, 0.5, 600],
    'LC':  [22.046, 18.143881, 0.025, 1, 400],
    'LRC': [1, 10, 1, 50, 10],
    'KW':  [0.3674333, 136.07911, 0.25, 10, 50],
    'SM':  [1.1023113, 90.718447, 0.1, 5, 100],
    'COM': [1.0604, 50, 0.25, 2.5, 53.02],
    'OBM': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}


def _compute_greeks(char, K, tau, vol, s, r, product, payoff, lots, ki=None, ko=None, barrier=None, direction=None):
    """ Computes the greeks of various option profiles. Currently, american and european greeks and pricing are assumed to be the same.

    Inputs:  1) char   : call or put
             2) K      : strike
             3) tau    : time to expiry
             4) vol    : volatility (sigma)
             5) s      : price of underlying
             6) r      : interest
             7) product: underlying commodity.
             8) payoff : american or european option.
             9) lots   : number of lots.
             10) barrier: american or european barrier.

    Outputs: 1) delta  : dC/dS
             2) gamma  : d^2C/dS^2
             3) theta  : dC/dt
             4) vega   : dC/dvol
    """

    # european options
    if tau == 0:
        print('tau == 0 case')
        gamma, theta, vega = 0, 0, 0
        if char == 'call':
            # in the money
            delta = 1 if K < s else 0
        if char == 'put':
            delta = -1 if K > s else 0
        return delta, gamma, theta, vega
    if payoff == 'euro' or payoff == 'amer':
        # vanilla case
        if barrier is None:
            # print('vanilla case')
            return _euro_vanilla_greeks(
                char, K, tau, vol, s, r, product, lots)
        elif barrier == 'amer':
            # print('amer barrier case')
            # greeks for european options with american barrier.
            return _euro_barrier_amer_greeks(char, tau, vol, K, s, r, payoff, direction, product, ki, ko, lots)
        elif barrier == 'euro':
            # print('euro barrier case')
            # greeks for european options with european barrier.
            return _euro_barrier_euro_greeks(char, tau, vol, K, s, r, payoff, direction, product, ki, ko, lots)


def _euro_vanilla_greeks(char, K, tau, vol, s, r, product, lots):
    """Summary

    Args:
        char (TYPE): Description
        K (TYPE): Description
        tau (TYPE): Description
        vol (TYPE): Description
        s (TYPE): Description
        r (TYPE): Description
        product (TYPE): Description

    Returns:
        TYPE: Description
    """
    # addressing degenerate case
    if vol == 0:
        gamma, theta, vega = 0, 0, 0
        if char == 'call':
            delta = 1 if K >= s else 0
        if char == 'put':
            delta = -1 if K >= s else 0
        return delta, theta, gamma, vega
    # print('VanInputs: ', char, K, tau, vol, s, r)
    d1 = (log(s/K) + (r + 0.5 * vol ** 2)*tau) / \
        (vol * sqrt(tau))
    d2 = d1 - vol*(sqrt(tau))

    # (1/sqrt(2*pi)) * exp(-(d1**2) / 2) / (s*vol*sqrt(tau))
    gamma1 = norm.pdf(d1)/(s*vol*sqrt(tau))
    vega1 = s * exp(r*tau) * norm.pdf(d1) * sqrt(tau)

    if char == 'call':
        # call option calc for delta and theta
        delta1 = norm.cdf(d1)
        theta1 = (-s * norm.pdf(d1)*vol) / (2*sqrt(tau))
    if char == 'put':
        # put option calc for delta and theta
        delta1 = norm.cdf(d1) - 1
        theta1 = (-s * norm.pdf(d1)*vol) / (2*sqrt(tau))

    delta, gamma, theta, vega = greeks_scaled(
        delta1, gamma1, theta1, vega1, product, lots)
    return delta, gamma, theta, vega


def greeks_scaled(delta1, gamma1, theta1, vega1, product, lots):
    return delta1, gamma1, theta1/365, vega1/100


def _euro_barrier_amer_greeks(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots, rebate=0):
    """Computes greeks of european options with american barriers. """
    ticksize = multipliers[product][2]
    change_spot = 0.1 * ticksize
    change_vol = 0.0001
    change_tau = 1/(24*365)
    # computing delta
    # char, tau, vol, k, s, r, payoff, direction, ki, ko, rebate=0
    del1 = _barrier_amer(char, tau, vol, k, s+change_spot,
                         r, payoff, direction, ki, ko)
    del2 = _barrier_amer(char, tau, vol,
                         k, max(0, s-change_spot),
                         r, payoff, direction, ki, ko)
    delta = (del1 - del2)/(2*change_spot)

    # computing gamma
    del3 = _barrier_amer(
        char, tau, vol, k, s, r, payoff, direction, ki, ko)
    gamma = (del1 - 2*del3 + del2)/(change_spot**2) if tau > 0 else 0

    # computing vega
    v1 = _barrier_amer(char, tau, vol+change_vol, k, s, r,
                       payoff, direction, ki, ko)
    tvol = max(0, vol - change_vol)

    v2 = _barrier_amer(char, tau, tvol, k, s, r,
                       payoff, direction, ki, ko)
    vega = (v1 - v2)/(2*change_vol) if tau > 0 else 0

    # computing theta
    t1 = _barrier_amer(char, tau, vol, k, s, r,
                       payoff, direction, ki, ko)
    t2 = _barrier_amer(char, tau-change_tau, vol, k, s, r,
                       payoff, direction, ki, ko)
    theta = (t2 - t1)/change_tau if tau > 0 else 0
    # scaling greeks to retrieve dollar value.
    delta, gamma, theta, vega = greeks_scaled(
        delta, gamma, theta, vega, product, lots)
    return delta, gamma, theta, vega
