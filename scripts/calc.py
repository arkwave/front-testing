"""
Script that contains implementation of the following calculation-related methods:

1) Pricing for various instruments:
    > vanilla options
    > barrier options

3) Calculating IV
4) Calculating PnL
 """

'''
TODO: 1) iv calculation [potentially not needed]
      2) barrier valuation <<<<<<< 3/9/2017
      3) option buying heuristics [to be discussed]
'''

from math import log, sqrt, exp, pi
from scipy.stats import norm


def _bsm(option, tau, vol, K, s, r):
    """Implementation of vanilla Black-Scholes Formula for Option Pricing.
    Inputs: 1) option     : call or put.
            2) tau        : time to expiry in years.
            3) vol        : volatility (sigma)
            4) K          : strike price
            5) underlying : price of underlying
            6) interest   : interest rate

    Output: 1) Price      : price of option according to BSM
    """
    d1 = (log(s/K) + (r + 0.5 * vol ** 2)*tau) / \
        (vol * sqrt(tau))
    d2 = d1 - vol*(sqrt(tau))
    nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
    negnd1, negnd2 = norm.cdf(-d1), norm.cdf(-d2)
    if option == 'call':
        price = nd1*s - nd2*K*exp(-r*tau)
    elif option == 'put':
        price = negnd2*K*exp(-r*tau) - negnd1*s
    return price


def _compute_greeks(char, K, tau, vol, s, r):
    """Closed-form computation of greeks.
    Inputs:  1) char   : call or put
             2) K      : strike
             3) tau    : time to expiry
             4) vol    : volatility (sigma)
             5) s      : price of underlying
             6) r      : interest

    Outputs: 1) delta  : dC/dS
             2) gamma  : d^2C/dS^2
             3) theta  : dC/dt
             4) vega   : dC/dvol
    """
    d1 = (log(s/K) + (r + 0.5 * vol ** 2)*tau) / \
        (vol * sqrt(tau))
    d2 = d1 - vol*(sqrt(tau))
    gamma = (1/sqrt(2*pi)) * exp(-(d1**2) / 2) / (s*vol*sqrt(tau))
    vega = s*(1/sqrt(2*pi)) * exp(-(d1**2) / 2) * sqrt(tau)
    if char == 'call':
        # call option calc for delta and theta
        delta = norm.cdf(d1)
        theta = ((-s * ((1/sqrt(2*pi)) * exp(-(d1**2) / 2)) * vol) /
                 2*sqrt(tau)) - (r * K * exp(-r*tau) * norm.cdf(d2))
    if char == 'put':
        # put option calc for delta and theta
        delta = norm.cdf(d1) - 1
        theta = ((-s * ((1/sqrt(2*pi)) * exp(-(d1**2) / 2)) * vol) /
                 2*sqrt(tau)) + (r * K * exp(-r*tau) * norm.cdf(-d2))
    return delta, gamma, theta, vega


def _compute_iv(underlying, price, strike, tau, r):
    pass


def _compute_pnl(portfolio):
    pass


def _compute_value(char, tau, vol, K, s, r, ki=None, ko=None):
    '''Wrapper function that computes value of option. 
    Inputs: 1) ki : Knock in value.
            2) ko : Knock out value.
            #) Remaining inputs are identical to _bsm inputs. '''
    # vanilla option case
    if ki is None and ko is None:
        return _bsm(char, tau, vol, K, s, r)
    # barrier option case
    else:
        return _barrier_valuation(char, tau, vol, K, s, r)


def _barrier_valuation(option, tau, vol, strike, underlying, interest):
    # implement barrier option valuations from Haug book.
    pass
