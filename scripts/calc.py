"""
File Name      : calc.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 23/3/2017
Python version : 3.5

Description:
Script contains implementation of the following calculation-related methods:

1) Pricing for various instruments:
    > vanilla options (american and european)
    > barrier options (american and european barriers)
    > computing greeks for all option classes.

2) Calculating implied volatility for European and American Options.

3) Various helper methods for numerical routines.


Notes:
1) Currently, European vanilla and American Vanilla are assumed to be valued the same/have the same greeks.
2) All barrier options require that the exercise structure be European. Even if an american option is passed in, it is valued like a European.

 """


from math import log, sqrt, exp, pi
from scipy.stats import norm
import numpy as np
import pandas as pd

# edf = pd.read_csv('datasets/cleaned_vol.csv').dropna()


# Dictionary of multipliers for greeks/pnl calculation.
# format  =  'product' : [dollar_mult, lot_mult, futures_tick,
# options_tick, pnl_mult]

# TODO: read this in during prep_data
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

filepath = 'portfolio_specs.txt'


#####################################################################
##################### Option pricing formulas #######################
#####################################################################


def _compute_value(char, tau, vol, K, s, r, payoff, ki=None, ko=None, barrier=None, d=None, product=None):
    '''Wrapper function that computes value of option.
    Inputs: 1) ki     : Knock in value.
            2) ko     : Knock out value.
            3) barrier: type of barrier
            4) d    : direction of barrier.
            #) Remaining inputs are identical to _bsm_euro.

    Outputs: Price of the option
    '''
    # vanilla option case
    if barrier is None:
        # currently american == european since it's never optimal to exercise
        # before expiry.
        if payoff == 'amer':
            return _bsm_euro(char, tau, vol, K, s, r)
        elif payoff == 'euro':
            return _bsm_euro(char, tau, vol, K, s, r)
    # barrier option case
    else:
        if barrier == 'amer':
            return _barrier_amer(char, tau, vol, K, s, r, payoff, d, ki, ko)
        elif barrier == 'euro':
            return _barrier_euro(char, tau, vol, K, s, r, payoff, d, ki, ko, product)


################### Vanilla Option Valuation ######################

def _bsm_euro(option, tau, vol, K, s, r):
    """Vanilla european option pricing.

    Inputs: 1) option     : call or put.
            2) tau        : time to expiry in years.
            3) vol        : volatility (sigma)
            4) K          : strike price
            5) underlying : price of underlying
            6) interest   : interest rate

    Output: 1) Price      : price of option according to BSM
    """
    if vol is None:
        raise ValueError('Vol cannot be None')
    elif vol == 0:
        val = max(s-K, 0) if option == 'call' else max(K-s, 0)
        return val
    d1 = (log(s/K) + (r + 0.5 * (vol ** 2))*tau) / \
        (vol * sqrt(tau))
    d2 = d1 - vol*(sqrt(tau))
    nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
    negnd1, negnd2 = norm.cdf(-d1), norm.cdf(-d2)
    if option == 'call':
        price = exp(-r*tau)*(nd1*s - nd2*K)
    elif option == 'put':
        price = exp(-r*tau)*(negnd2*K - negnd1*s)
    return price


def _amer_option(option, tau, vol, K, s, r):
    """Vanilla american option pricing.

    Inputs: 1) option     : call or put.
            2) tau        : time to expiry in years.
            3) vol        : volatility (sigma)
            4) K          : strike pricef
            5) underlying : price of underlying
            6) interest   : interest rate

    Output: 1) Price      : price of option according to CRR Binomial Tree
    """
    return _CRRbinomial('price', 'amer', option, s, k, tau, r, r, vol)


###########################################################################

####################### Barrier Option Valuation ##########################

def get_barrier_vol(df, product, tau, call_put_id, barlevel):
    df['product'] = df['vol_id'].str.split().str[0]
    bvol = df[(np.isclose(df['tau'], tau, atol=1e-4)) &
              (df['call_put_id'] == call_put_id) &
              (df['strike'] == barlevel) &
              (df['product'] == product)]
    # print('BVOL BEFORE: ', bvol)
    bvol = bvol['settle_vol'].values[0]
    # print('BVOL: ', bvol)
    return bvol


def _barrier_euro(char, tau, vol, k, s, r, payoff, direction, ki, ko, product, rebate=0):
    from .prep_data import read_data
    voldf, pricedf, edf = read_data(filepath)
    """ Pricing model for options with European barriers.

    Inputs:
    1) Char      : call or put.
    2) tau       : time to expiry.
    3) vol       : volatility
    4) k         : strike price
    5) s         : price of underlying
    6) r         : interest rate
    7) payoff    : european or american option
    8) direction : direction of the barrier.
    9) rebate    : premium returned if option knocks out. currently defaulted to 0
    10) ki       : knock in barrier amount.
    11) ko       : knock out barrier amount.

    Outputs:
    1) Price

    """
    # check for expiry
    # if tau == 0:
    #     return _compute_value(char, tau, vol, K, s, r, payoff)
    barlevel = ki if ki else ko
    call_put_id = 'C' if char == 'call' else 'P'
    bvol = get_barrier_vol(voldf, product, tau, call_put_id, barlevel)
    # case when barrier vol is not in vol surface; raise error.
    if bvol is None:
        raise ValueError('Improper Data: Barrier vol not on vol surface.')
    ticksize = multipliers[product][2]
    if ki:
        calc_lots = (k - ki)/ticksize
    if ko:
        calc_lots = (k - ko)/ticksize
    if char == 'call':
        if direction == 'up':
            if ki:
                # call up in
                return _compute_value(char, tau, vol, k, s, r, payoff)
            if ko:
                # call up out
                p1 = call_put_spread(
                    s, ko, k, r, bvol, vol, tau, 'callspread', payoff)
                p2 = call_put_spread(
                    s, ko, ko-ticksize, r, bvol, bvol, tau, 'callspread', payoff)
                return p1 - calc_lots*p2
        if direction == 'down':
            if ki:
                # call down in
                p1 = call_put_spread(s, ki, k, r, bvol, vol, tau,
                                     'callspread', payoff)
                p2 = call_put_spread(
                    s, ki + ticksize, ki, r, bvol, bvol, tau, 'callspread', payoff)
                return p1 - calc_lots * p2

            if ko:
                # call down out
                return _compute_value(char, tau, vol, k, s, r, payoff)

    if char == 'put':
        if direction == 'up':
            if ki:
                # put up in
                p1 = call_put_spread(
                    s, k, ki, r, vol, bvol, tau, 'putspread', payoff)
                p2 = call_put_spread(
                    s, ki, ki-ticksize, r, bvol, bvol, tau, 'putspread', payoff)
                return p1 - calc_lots*p2
            if ko:
                return _compute_value(char, tau, vol, k, s, r, payoff)
        if direction == 'down':
            if ki:
                # put down in
                return _compute_value(char, tau, vol, k, s, r, payoff)

            if ko:
                # put down out
                p1 = call_put_spread(
                    s, k, ko, r, vol, bvol, tau, 'putspread', payoff)
                p2 = call_put_spread(
                    s, ko + ticksize, ko, r, bvol, bvol, tau, 'putspread', payoff)
                return p1 - calc_lots*p2
    return price


def _barrier_amer(char, tau, vol, k, s, r, payoff, direction, ki, ko, rebate=0):
    """ Pricing model for options with american barrers. Currently, payoff is assumed to be European; consequently _compute_value defaults to computing the value of a European vanilla option.

    Inputs:
    1) Char      : call or put.
    2) tau       : time to expiry.
    3) vol       : volatility
    4) k         : strike price
    5) s         : price of underlying
    6) r         : interest rate
    7) payoff    : european or american option
    8) direction : direction of the barrier.
    9) rebate    : premium returned if option knocks out.
    10) ki       : knock in barrier amount.
    11) ko       : knock out barrier amount.

    Outputs:
    1) Price

    """
    # dealing with expiry case.
    # if tau == 0:
    #     return _compute_value(char, tau, vol, k, s, r, payoff)

    # initializing constants
    # scaling for calc
    eta = -1 if direction == 'up' else 1
    phi = 1 if char == 'call' else -1
    b = 0
    mu = (b - ((vol**2)/2))/(vol**2)
    lambd = sqrt(mu**2 + 2*r/vol**2)
    h = ki if ki else ko

    x1 = log(s/k)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
    x2 = log(s/h)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
    y1 = log(h**2/(s*k))/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
    y2 = log(h/s)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
    z = log(h/s)/(vol * sqrt(tau)) + lambd*(vol*sqrt(tau))

    A = A_B('A', phi, b, r, x1, x2, tau, vol, s, k)
    B = A_B('B', phi, b, r, x1, x2, tau, vol, s, k)
    C = C_D('C', phi, s, b, r, tau, h, mu, eta, y1, y2, k, vol)
    D = C_D('D', phi, s, b, r, tau, h, mu, eta, y1, y2, k, vol)
    E = E_f(rebate, r, tau, eta, x2, vol, h, s, mu, y2)
    F = F_f(rebate, h, s, mu, lambd, eta, z, vol, tau)

    # pricing logic

    # call options
    if char == 'call':
        if direction == 'up':
            # call up in
            if ki:
                if s >= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s < ki and k >= ki and tau > 0:
                    return A + E
                elif s < ki and k >= ki and tau == 0:
                    return 0
                elif s < ki and k < ki and tau > 0:
                    return B - C + D + E
                elif s < ki and k < ki and tau == 0:
                    return 0
            # call_up_out
            if ko:
                if s >= ko:
                    return rebate * exp(-r*tau)
                elif s < ko and k >= ko and tau > 0:
                    return F
                elif s < ko and k >= ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s < ko and k < ko and tau > 0:
                    return A - B + C - D + F
                elif s < ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)

        if direction == 'down':
            if ki:
                # call_down_in
                if s <= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s > ki and k >= ki and tau > 0:
                    return C + E
                elif s > ki and k >= ki and tau == 0:
                    return 0
                elif s > ki and k < ki and tau > 0:
                    return A - B + D + E
                elif s > ki and k < ki and tau == 0:
                    return 0
            if ko:
                # call_down_out
                if s < ko:
                    return rebate*exp(-r*tau)
                elif s > ko and k >= ko and tau > 0:
                    return A - C + F
                elif s > ko and k >= ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s > ko and k < ko and tau > 0:
                    return B - D + F
                elif s > ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)

    # put options
    elif char == 'put':
        if direction == 'up':
            if ki:
                # put_up_in
                if s >= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s < ki and k >= ki and tau > 0:
                    return A - B + D + E
                elif s < ki and k >= ki and tau == 0:
                    return 0
                elif s < ki and k < ki and tau > 0:
                    return C + E
                elif s < ki and k < ki and tau == 0:
                    return 0
            if ko:
                # put_up_out
                if s >= ko:
                    return rebate * exp(-r*tau)
                elif s < ko and k >= ko and tau > 0:
                    return B - D + F
                elif s < ko and k >= ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s < ko and k < ko and tau > 0:
                    return A - C + F
                elif s < ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)

        if direction == 'down':
            if ki:
                # put_down_in
                if s <= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s > ki and k >= ki and tau > 0:
                    return B - C + D + E
                elif s > ki and k >= ki and tau == 0:
                    return 0
                elif s > ki and k < ki and tau > 0:
                    return A + E
                elif s > ki and k < ki and tau == 0:
                    return 0

            if ko:
                # put_down_out
                if s <= ko:
                    return rebate * exp(-r*tau)
                elif s > ko and k > ko and tau > 0:
                    return A - B + C - D + F
                elif s > ko and k > ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)
                elif s > ko and k < ko and tau > 0:
                    return F
                elif s > ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff)

##########################################################################


########################## Call-Put Spread Valuation #####################
def call_put_spread(s, k1, k2, r, vol1, bvol, tau, optiontype, payoff, b=0):
    # call spread
    # _compute_value(char, tau, vol, K, s, r, payoff, product, ki=None,
    # ko=None, barrier=None, d=None)
    price = 0
    if optiontype == 'callspread':
        p1 = _compute_value('call', tau, vol1, k1, s, r, payoff)
        p2 = _compute_value('call', tau, bvol, k2, s, r, payoff)
        price = p2 - p1
    elif optiontype == 'putspread':
        p1 = _compute_value('put', tau, vol1, k1, s, r, payoff)
        p2 = _compute_value('put', tau, bvol, k2, s, r, payoff)
        price = p1 - p2
    return price


def call_put_spread_greeks(s, k1, k2, r, vol1, bvol, tau, optiontype, product, lots, payoff, b=0):
    if optiontype == 'callspread':
        try:
            delta1, gamma1, theta1, vega1 = _compute_greeks(
                'call', k1, tau, vol1, s, r, product, payoff, lots)
            delta2, gamma2, theta2, vega2 = _compute_greeks(
                'call', k2, tau, bvol, s, r, product, payoff, lots)
            delta = delta2 - delta1
            gamma = gamma2 - gamma1
            vega = vega2 - vega1
            theta = theta2 - theta1
        except TypeError:
            print('tau: ', tau)
            print('k1: ', k1)
            print('k2: ', k2)
            print('vol1: ', vol1)
            print('bvol: ', bvol)
            print('s: ', s)
            print('r: ', r)
            print('product: ', product)
            print('payoff: ', payoff)
            print('op1: ', _compute_greeks(
                'call', k1, tau, vol1, s, r, product, payoff, lots))
            print('op2: ', _compute_greeks(
                'call', k2, tau, bvol, s, r, product, payoff, lots))
    elif optiontype == 'putspread':
        delta1, gamma1, theta1, vega1 = _compute_greeks(
            'put', k1, tau, vol1, s, r, product, payoff, lots)
        delta2, gamma2, theta2, vega2 = _compute_greeks(
            'put', k2, tau, bvol, s, r, product, payoff, lots)
        delta = -(delta2 - delta1)
        gamma = -(gamma2 - gamma1)
        vega = -(vega2 - vega1)
        theta = -(theta2 - theta1)
    return delta, gamma, theta, vega
#############################################################################


#############################################################################
##################### Greek-related formulas ################################
#############################################################################
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

    # # american options
    # elif payoff == 'amer':
    #     # vanilla case
    #     if barrier is None:
    #         return _amer_vanilla_greeks(
    #             char, K, tau, vol, s, r, product, lots)
    #     # american options with american barrier
    #     elif barrier == 'amer':
    #         return _amer_barrier_amer_greeks()
    #     # american option with european barrier.
    #     elif barrier == 'euro':
    #         return _amer_barrier_euro_greeks()


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


# NIU: Not in Use.
def _amer_vanilla_greeks(char, K, tau, vol, s, r, product, lots):
    """Computes greeks for vanilla American options.

    Args:
        char (str)    : call or put.
        K (double)    : strike
        tau (double)  : time to expiry
        vol (double)  : volatility
        s (double)    : price of underlying
        r (double)    : interest rate
        product (str) : underlying commodity name.

    Returns:
        TYPE: Description
    """
    delta, gamma, theta, vega = _CRRbinomial(
        'greeks', 'amer', char, s, K, tau, r, r, vol, product)
    delta, gamma, theta, vega = greeks_scaled(
        delta, gamma, theta, vega, product, lots)
    return delta, gamma, theta, vega


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


def _euro_barrier_euro_greeks(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots, rebate=0):
    """Computes greeks of european options with european barriers. """
    from .prep_data import read_data
    voldf, pricedf, edf = read_data(filepath)
    call_put_id = 'C' if char == 'call' else 'P'
    barlevel = ki if ki else ko
    ticksize = multipliers[product][2]
    bvol = get_barrier_vol(voldf, product, tau, call_put_id, barlevel)
    if ki:
        calc_lots = (k - ki)/ticksize
    if ko:
        calc_lots = (k - ko)/ticksize
    if char == 'call':
        if direction == 'up':
            if ki:
                # call up in
                return _compute_greeks(char, tau, vol, k, s, r, product, payoff, lots)
            if ko:
                # call up out
                d1, g1, t1, v1 = call_put_spread_greeks(
                    s, ko, k, r, bvol, vol, tau, 'callspread', product, lots, payoff)
                d2, g2, t2, v2 = call_put_spread_greeks(
                    s, ko, ko-ticksize, r, bvol, bvol, tau, 'callspread', product, lots, payoff)
                delta = d1 - calc_lots*d2
                gamma = g1 - calc_lots*g2
                theta = t1 - calc_lots*t2
                vega = v1 - calc_lots*v2
                return delta, gamma, theta, vega
        if direction == 'down':
            if ki:
                # call down in
                d1, g1, t1, v1 = call_put_spread_greeks(s, ki, k, r, bvol, vol, tau,
                                                        'callspread', product, lots, payoff)
                d2, g2, t2, v2 = call_put_spread_greeks(
                    s, ki + ticksize, ki, r, bvol, bvol, tau, 'callspread', product, lots, payoff)
                delta = d1 - calc_lots*d2
                gamma = g1 - calc_lots*g2
                theta = t1 - calc_lots*t2
                vega = v1 - calc_lots*v2
                return delta, gamma, theta, vega
            if ko:
                # call down out
                return _compute_greeks(char, tau, vol, k, s, r, product, payoff, lots)

    if char == 'put':
        if direction == 'up':
            if ki:
                # put up in
                d1, g1, t1, v1 = call_put_spread_greeks(
                    s, k, ki, r, vol, bvol, tau, 'putspread', product, lots, payoff)
                d2, g2, t2, v2 = call_put_spread_greeks(
                    s, ki, ki-ticksize, r, bvol, bvol, tau, 'putspread', product, lots, payoff)
                delta = d1 - calc_lots*d2
                gamma = g1 - calc_lots*g2
                theta = t1 - calc_lots*t2
                vega = v1 - calc_lots*v2
                return delta, gamma, theta, vega
            if ko:
                return _compute_greeks(char, tau, vol, k, s, r, product, payoff, lots)
        if direction == 'down':
            if ki:
                # put down in
                return _compute_greeks(char, tau, vol, k, s, r, product, payoff, lots)
            if ko:
                # put down out
                d1, g1, t1, v1 = call_put_spread_greeks(
                    s, k, ko, r, vol, bvol, tau, 'putspread', product, lots, payoff)
                d2, g2, t2, v2 = call_put_spread_greeks(
                    s, ko + ticksize, ko, r, bvol, bvol, tau, 'putspread', product, lots, payoff)

                delta = d1 - calc_lots*d2
                gamma = g1 - calc_lots*g2
                theta = t1 - calc_lots*t2
                vega = v1 - calc_lots*v2
                return delta, gamma, theta, vega

# def _amer_barrier_euro_greeks():
#     """Computes greeks of american options with european barriers. """
#     pass


# def _amer_barrier_amer_greeks(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, rebate=0):
#     """Computes greeks of american options with american barriers. """
#     pass


def _compute_iv(optiontype, s, k, c, tau, r, flag):
    """Computes implied volatility of plain vanilla puts and calls.
    Inputs:
    1) optiontype : call or put.
    2) s          : underlying
    3) k          : strike
    4) c          : option price
    5) tau        : time to expiry
    6) flag       : american or european option
    7) product    : underlying product"""

    # european option
    if flag == 'euro':
        return newton_raphson(optiontype, s, k, c, tau, r)
    # american option
    elif flag == 'amer':
        return newton_raphson(optiontype, s, k, c, tau, r)
        # return american_iv(s, k, c, tau, r, optiontype)


####################################################################
##### Supplemental Methods for Numerical Approximations ############
####################################################################

def newton_raphson(option, s, k, c, tau, r, num_iter=100):
    """Newton's method for calculating IV for vanilla european options.
    Inputs:
    1) option  : call or put.
    2) s       : underlying price
    3) k       : strike
    4) c       : option price
    5) tau     : time to expiry
    6) r       : interest rate
    7) num_iter: number of iterations to run the numerical procedure. defaults to 100.

    Outputs: Implied volatility of this VANILLA EUROPEAN option.
     """
    precision = 1e-3
    guess = sqrt(2*pi / tau)*(c/s)
    for i in range(num_iter):
        try:
            d1 = (log(s/k) + (r + 0.5 * guess ** 2)*tau) / \
                (guess * sqrt(tau))
            d11 = (guess**2 * tau * sqrt(tau) -
                   (log(s/k)+(r+guess**2)*tau)*sqrt(tau)) / (guess**(2*tau))
            option_price = _bsm_euro(option, tau, guess, k, s, r)
            vega = s * norm.pdf(d1) * sqrt(tau)
            pvega = s * norm.pdf(d1) * d11 - k*exp(-r*tau)
            # vega = s*(1/sqrt(2*pi)) * exp(-(d1**2) / 2) * sqrt(tau)
            diff = option_price - c
            if abs(diff) < precision:
                return guess
            guess = guess - diff/vega
        except RuntimeWarning:
            print('guess: ', guess)
            print('diff: ', diff)
            print('tau: ', tau)
            print('vol: ', vol)
    if np.isnan(guess):
        guess = 0
    return guess


def greeks_scaled(delta1, gamma1, theta1, vega1, product, lots):
    """Summary

    Args:
        delta1 (TYPE): Description
        gamma1 (TYPE): Description
        theta1 (TYPE): Description
        vega1 (TYPE): Description
        product (TYPE): Description
        lots (TYPE): Description

    Returns:
        TYPE: Description
    """

    # lots = lots
    # lm = multipliers[product][1]
    # dm = multipliers[product][0]
    # delta = delta1 * lots
    # gamma = gamma1*lots*lm/dm
    # vega = vega1*lots*lm*dm/100
    # theta = theta1*lots*lm*dm/365

    return delta1, gamma1, theta1/365, vega1/100
    # return delta, gamma, theta, vega


# NIU: not currently being used.
def american_iv(option, s, k, c, tau, r, product, num_iter=100):
    """Newton's method for calculating IV for vanilla european options.
    Inputs:
    1) optiontype  : call or put.
    2) s           : underlying price
    3) k           : strike
    4) c           : option price
    5) tau         : time to expiry
    6) r           : interest rate

    Outputs: Implied volatility of this VANILLA AMERICAN option.
    """
    precision = 1e-5
    guess = 0.5
    for i in range(num_iter):
        try:
            d1 = (log(s/K) + (r + 0.5 * guess ** 2)*tau) / \
                (guess * sqrt(tau))
            option_price = _amer_option(option, tau, guess, K, s, r, product)
            vega = _num_vega('amer', option, s, k, tau, r, r, guess)
            price = option_price
            diff = c - option_price
            if abs(diff) < precision:
                return sigma
            sigma = sigma + diff/vega
        except RuntimeWarning:
            print('guess: ', guess)
            print('diff: ', diff)
            print('tau: ', tau)
            print('vol: ', vol)
    return sigma


# NIU: Not in Use.
def _CRRbinomial(output_flag, payoff, option_type, s, k, tau, r, vol, product, n=100, b=0):
    """Implementation of a Cox-Ross-Rubinstein Binomial tree. Translalted from The Complete Guide to Options
    Pricing Formulas, Chapter 7: Trees and Finite Difference methods by Espen Haug.

    Inputs:
        output_flag   : 'greeks' or 'price'
        payoff        : 'amer' or 'euro'
        option_type   : 'call' or 'put'
        s             : price of underlying
        k             : strike price
        tau           : time to expiry
        r             : interest rate
        b             : cost of carry. defaulted to 0
        vol           : volatility
        product       : the underlying product.
        n             : number of timesteps. defaults to 100.

    Returns:
        Price  : the price of this option
        Greeks : returns delta, gamma, theta, vega
    """

    if option_type == 'call':
        z = 1
    else:
        z = -1
    # list containing values to be returned.
    returnvalue = [0, 0, 0, 0]
    optionvalue = []             # list containing values of option at diff t
    dt = tau/n                   # timestep
    u = exp(vol * sqrt(dt))      # increase value
    d = 1/u                      # decrease value
    p = (exp(b*dt) - d) / (u-d)  # probability of increase
    Df = exp(-r*dt)

    for i in range(0, n):
        optionvalue[i] = max(0, z*(s * u**i * d**(n-i) - k))

    for j in range(n-1, 0, -1):
        for i in range(0, j):
            if payoff == 'euro':
                optionvalue[i] = (p*optionvalue[i+1] + (1-p)*optionvalue[i])*Df
            elif payoff == 'amer':
                # check for max between exercise and holding.
                optionvalue[i] = max(
                    (z * (s * (u**i) * d**(j-i)-k)), ((p*optionvalue[i+1] + (1-p)*optionvalue[i])*Df))
        if j == 2:
            returnvalue[2] = ((optionvalue[2] - optionvalue[1])/(s*u**2 - s)) - (
                optionvalue[1] - optionvalue[0])/(s-s*d**2) / (0.5 * (s * u**2 - s * d**2))
            returnvalue[3] = optionvalue[1]
        if j == 1:
            returnvalue[1] = (optionvalue[1] - optionvalue[0])/(s * u - s * d)

    returnvalue[3] = (returnvalue[3] - optionvalue[0]) / (2*dt) / 365
    returnvalue[0] = optionvalue[0]

    if output_flag == 'price':
        price = returnvalue[0]
        return price
    else:
        greeks = returnvalue[1:]
        vega = _num_vega(
            output_flag, payoff, option_type, s, k, tau, r, vol)
        greeks.append(vega)
        delta, gamma, theta, vega = greeks[0], greeks[1], greeks[2], greeks[3]
        return delta, gamma, theta, vega


# NIU: Not in Use.
def _num_vega(payoff, option_type, s, k, tau, r,  vol, b=0):
    """Computes vega from CRR binomial tree if option passed in is American, and uses
     closed-form analytic formula for vega if option is European.

    Args:
        payoff (string)     : type of option. 'amer' or 'euro'.
        option_type (string): 'call' or 'put'
        s (double)          : price of underlying
        k (double)          : strike
        tau (double)        : time to expiry
        r (double)          : interest rate
        b (double)          : cost of carry
        vol (double)        : volatility

    Returns:
        vega (double)
    """
    delta_v = 0.01
    # american case. use finite difference approach to approximate local vega
    if payoff == 'amer':
        upper = _CRRbinomial(
            'price', payoff, option_type, s, k, tau, r, vol+delta_v)
        lower = _CRRbinomial(
            'price', payoff, option_type, s, k, tau, r, vol-delta_v)
        vega = (upper - lower)/(2*delta_v)
    # european case. use analytic formulation for vega.
    elif payoff == 'euro':
        vega = s*(1/sqrt(2*pi)) * exp(-(d1**2) / 2) * sqrt(tau)

    # vega multiplier
    lots = lots
    lm = multipliers[product][1]
    dm = multipliers[product][0]
    vega = vega1*lots*lm*dm/100

    return vega


####################################################################
########### Barrier Option Valuation Helper Methods ################
####################################################################

# Note: the following are taken from Haug: The Complete Guide to Option Pricing


def A_B(flag, phi, b, r, x1, x2, tau, vol, s, k):
    # vol = vol*100
    x = x1 if flag == 'A' else x2
    ret = phi*s*exp((b-r)*tau)*norm.cdf(phi*x) - phi*k * \
        exp(-r*tau)*norm.cdf(phi*x - phi*vol*sqrt(tau))
    return ret


def C_D(flag, phi, s, b, r, tau, h, mu, eta, y1, y2, k, vol):
    # vol = vol*100
    y = y1 if flag == 'C' else y2
    ret = (phi*s*exp((b-r)*tau)*((h/s)**(2*(mu + 1)))*norm.cdf(eta*y)) - \
        phi*k*exp(-r*tau) * ((h/s)**(2*mu))*norm.cdf(eta*y - eta*vol*sqrt(tau))

    # ret =  phi*s*exp((b-r)*tau)*((h/s)**(2*(1 + mu)))*norm.cdf(eta*y) - \
    # phi*k*exp(-r*tau) * ((h/s) ^ (2*mu))*norm.cdf(eta*y - eta*vol*sqrt(tau))
    return ret


def E_f(k, r, tau, eta, x, vol, h, s, mu, y):
    # tau = tau * 365
    # vol = vol*100
    ret = k*exp(-r*tau) * (norm.cdf(eta*x - eta*vol*sqrt(tau)) -
                           ((h/s)**(2*mu)) * norm.cdf(eta*y - eta*vol*sqrt(tau)))
    # ret = k*exp(-r*tau) * (norm.cdf(eta*x - eta*vol*sqrt(tau)) -
    #                         ((h/s)^(2*mu))*  norm.cdf(eta*y - eta*vol*sqrt(tau)))
    return ret


def F_f(k, h, s, mu, l, eta, z, vol, tau):
    # vol = vol*100
    ret = k * (((h/s)**(mu + l)) * norm.cdf(eta*z) + (h/s)**(mu-l)
               * norm.cdf(eta*z - 2*eta*l * vol*sqrt(tau)))
    return ret
