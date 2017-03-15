"""
File Name      : calc.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 15/3/2017
Python version : 3.5
Description:

Script contains implementation of the following calculation-related methods:

1) Pricing for various instruments:
    > vanilla options (american and european)
    > barrier options (american and european barriers)
    > computing greeks for all option classes.

2) Calculating implied volatility for European and American Options.

3) Various helper methods for numerical routines.

 """

# format :
# 'product' : [dollar_mult, lot_mult, futures_tick, options_tick, pnl_mult]

multipliers =
{
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

# product : [futures_multiplier - (dollar_mult, lot_mult), futures_tick,
# options_tick, pnl mult]

from math import log, sqrt, exp, pi
from scipy.stats import norm


#####################################################################
##################### Option pricing formulas #######################
#####################################################################


def _compute_value(char, tau, vol, K, s, r, payoff, product, ki=None, ko=None, barrier=None, d=None):
    '''Wrapper function that computes value of option.
    Inputs: 1) ki     : Knock in value.
            2) ko     : Knock out value.
            3) barrier: type of barrier
            4) d    : direction of barrier.
            #) Remaining inputs are identical to _bsm inputs.

    Outputs: Price of the option
    '''

    # vanilla option case
    if ki is None and ko is None:
        if payoff == 'amer':
            return _amer_option(char, tau, vol, K, s, r, product)
        elif payoff == 'euro':
            return _bsm_euro(char, tau, vol, K, s, r, product)
    # barrier option case
    else:
        if barrier == 'amer':
            return _barrier_amer(char, tau, vol, K, s, r, payoff, d, product, ki, ko)
        elif barrier == 'euro':
            return _barrier_euro(char, tau, vol, K, s, r, payoff, d, product, ki, ko)


################### Vanilla Option Valuation ######################

def _bsm_euro(option, tau, vol, K, s, r, product):
    """Vanilla european option pricing.

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
        price = exp(-r*tau)*(nd1*s - nd2*K)
    elif option == 'put':
        price = exp(-r*tau)*(negnd2*K - negnd1*s)
    return price


def _amer_option(option, tau, vol, K, s, r, product):
    """Vanilla american option pricing.

    Inputs: 1) option     : call or put.
            2) tau        : time to expiry in years.
            3) vol        : volatility (sigma)
            4) K          : strike price
            5) underlying : price of underlying
            6) interest   : interest rate

    Output: 1) Price      : price of option according to CRR Binomial Tree
    """
    return _CRRbinomial('price', 'amer', option, s, k, tau, r, r, vol, product)


###########################################################################

####################### Barrier Option Valuation ##########################


def _barrier_euro(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, rebate=0):
    """ Pricing model for options with european barrers.

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
    pass


def _barrier_amer(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, rebate):
    """ Pricing model for options with american barrers.

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
    # initializing constants
    eta = -1 if direction == 'up' else 1
    phi = 1 if char == 'call' else -1
    b = 0
    mu = (b - ((vol**2)/2))/(vol**2)
    lambd = sqrt(mu + 2*r/vol**2)
    x1 = log(s/k)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
    if ki:
        x2 = log(s/ki)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
        y1 = log(ki**2/(s*k))/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
        y2 = log(ki/s)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
        z = log(ki/s)/(vol * sqrt(tau)) + lambd*(vol*sqrt(tau))
    if ko:
        x2 = log(s/ko)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
        y1 = log(ko**2/(s*k))/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
        y2 = log(ko/s)/(vol * sqrt(tau)) + (1 + mu)*vol*sqrt(tau)
        z = log(ko/s)/(vol * sqrt(tau)) + lambd*(vol*sqrt(tau))

    A = A_B('A', phi, b, r, x1, x2, tau, vol, s, k)
    B = A_B('B', phi, b, r, x1, x2, tau, vol, s, k)
    C = C_D('C', phi, s, b, r, t, h, mu, eta, y1, y2, k, vol)
    D = C_D('D', phi, s, b, r, t, h, mu, eta, y1, y2, k, vol)
    E = E(k, r, tau, eta, x2, vol, h, s, mu, y2)
    F = F(k, h, s, mu, l, eta, z, vol, tau)

    # pricing logic

    # option has expired.

    # call options
    if char == 'call':
        if direction == 'up':
            if ki:
                if tau == 0:
                    return 0
                if s >= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
                elif s < ki:
                    if k >= ki:
                        return A + E
                    if k < ki:
                        return B - C + D + E
            # call_up_out
            if ko:
                if s >= ko:
                    return rebate * exp(-r*tau)
                elif s < ko and k >= ko and tau > 0:
                    return F
                elif s < ko and k >= ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
                elif s < ko and k < ko and tau > 0:
                    return A - B + C - D + F
                elif s < ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)

        if direction == 'down':
            if ki:
                # call_down_in
                if s <= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
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
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
                elif s > ko and k < ko and tau > 0:
                    return B - D + F
                elif s > ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)

    # put options
    elif char == 'put':
        if direction == 'up':
            if ki:
                # put_up_in
                if s >= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
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
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
                elif s < ko and k < ko and tau > 0:
                    return A - C + F
                elif s < ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)

        if direction == 'down':
            if ki:
                # put_down_in
                if s <= ki:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
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
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)
                elif s > ko and k < ko and tau > 0:
                    return F
                elif s > ko and k < ko and tau == 0:
                    return _compute_value(char, tau, vol, k, s, r, payoff, product)

##########################################################################


#####################################################################
##################### Greek-related formulas ########################
#####################################################################


def _compute_greeks(char, K, tau, vol, s, r, product, payoff, lots, barrier=None):
    """ Computes the greeks of various option profiles.
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
    if payoff == 'euro':
        # vanilla case
        if barrier is None:
            return _euro_vanilla_greeks(
                char, K, tau, vol, s, r, product, lots)
        elif barrier == 'amer':
            # greeks for european options with american barrier.
            return _euro_barrier_amer_greeks()
        elif barrier == 'euro':
            # greeks for european options with european barrier.
            return _euro_barrier_euro_greeks()

    # american options
    elif payoff == 'amer':
        # vanilla case
        if barrier is None:
            return _amer_vanilla_greeks(
                char, K, tau, vol, s, r, product, lots)
        # american options with american barrier
        elif barrier == 'amer':
            return _amer_barrier_amer_greeks()
        # american option with european barrier.
        elif barrier == 'euro':
            return _amer_barrier_euro_greeks()


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

    d1 = (log(s/K) + (r + 0.5 * vol ** 2)*tau) / \
        (vol * sqrt(tau))
    d2 = d1 - vol*(sqrt(tau))

    gamma1 = (1/sqrt(2*pi)) * exp(-(d1**2) / 2) / (s*vol*sqrt(tau))
    vega1 = s*(1/sqrt(2*pi)) * exp(-(d1**2) / 2) * sqrt(tau)

    if char == 'call':
        # call option calc for delta and theta
        delta1 = norm.cdf(d1)
        theta1 = ((-s * ((1/sqrt(2*pi)) * exp(-(d1**2) / 2)) * vol) /
                  2*sqrt(tau)) - (r * K * exp(-r*tau) * norm.cdf(d2))
    if char == 'put':
        # put option calc for delta and theta
        delta1 = norm.cdf(d1) - 1
        theta1 = ((-s * ((1/sqrt(2*pi)) * exp(-(d1**2) / 2)) * vol) /
                  2*sqrt(tau)) + (r * K * exp(-r*tau) * norm.cdf(-d2))
    delta, gamma, theta, vega = greeks_scaled(
        delta1, gamma1, theta1, vega1, product, lots)
    return delta, gamma, theta, vega


def _amer_vanilla_greeks(char, K, tau, vol, s, r, product):
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


# TODO: _euro_barrier_amer_g()
def _euro_barrier_amer_greeks():
    """Computes greeks of european options with american barriers. """
    pass

# TODO: _euro_barrier_euro_g()


def _euro_barrier_euro_greeks():
    """Computes greeks of european options with european barriers. """
    pass


# TODO: _amer_barrier_euro_g
def _amer_barrier_euro_greeks():
    """Computes greeks of american options with european barriers. """
    pass


# TODO: _amer_barrier_amer_g
def _amer_barrier_amer_greeks():
    """Computes greeks of american options with american barriers. """
    pass


def _compute_iv(optiontype, s, k, c, tau, r, flag, product):
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
        return newton_raphson(s, k, c, tau, r, optiontype)
    # american option
    elif flag == 'amer':
        return american_iv(s, k, c, tau, r, optiontype)


####################################################################
##### Supplemental Methods for Numerical Approximations ############
####################################################################

def newton_raphson(option, s, k, c, tau, r, product, num_iter=100):
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
    precision = 1e-5
    guess = 0.5
    for i in range(num_iter):
        d1 = (log(s/K) + (r + 0.5 * guess ** 2)*tau) / \
            (guess * sqrt(tau))
        option_price = _bsm_euro(option, tau, guess, k, s, r, product)
        vega = s*(1/sqrt(2*pi)) * exp(-(d1**2) / 2) * sqrt(tau)
        price = option_price
        diff = c - option_price
        if abs(diff) < precision:
            return sigma
        sigma = sigma + diff/vega
    return sigma


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
        d1 = (log(s/K) + (r + 0.5 * guess ** 2)*tau) / \
            (guess * sqrt(tau))
        option_price = _amer_option(option, tau, guess, K, s, r, product)
        vega = _num_vega('amer', option, s, k, tau, r, r, guess)
        price = option_price
        diff = c - option_price
        if abs(diff) < precision:
            return sigma
        sigma = sigma + diff/vega
    return sigma


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
    lots = lots
    lm = multipliers[product][1]
    dm = multipliers[product][0]
    delta = delta1 * lots
    gamma = gamma1*lots*lm/dm
    vega = vega1*lots*lm*dm/100
    theta = theta1*lots*lm*dm

    return delta, gamma, theta, vega

####################################################################
#### Barrier Option Valuation Helper Methods #######################
####################################################################

# Note: the following are taken from Haug: The Complete Guide to Option Pricing


def A_B(flag, phi, b, r, x1, x2, tau, vol, s, k):
    x == x1 if flag == 'A' else x2
    ret = phi*s*exp((b-r)*tau)*norm.cdf(phi*x) - phi*k * \
        exp(-r*tau)*norm.cdf(phi*x - phi*vol*sqrt(tau))
    return ret


def C_D(flag, phi, s, b, r, t, h, mu, eta, y1, y2, k, vol):
    y = y1 if flag == 'C' else y2
    ret = (phi*s*exp((b-r)*tau) * (h/s)**(2*mu + 1) * norm.cdf(eta*y)) - \
        (phi*k*exp(-r*tau) * (h/s)**(2*mu)
         * norm.cdf(eta*y1 - eta*vol*sqrt(tau)))
    return ret


def E(k, r, tau, eta, x, vol, h, s, mu, y):
    ret = k*exp(-r*tau) * ((norm.cdf(eta*x - eta*vol*sqrt(tau))) -
                           ((h/s)**(2*mu) * norm.cdf(eta*y - eta*vol*sqrt(tau))))
    return ret


def F(k, h, s, mu, l, eta, z, vol, tau):
    ret = k * (((h/s)**(mu + l) * norm.cdf(eta*z)) +
               ((h/s)**(mu-l)*norm.cdf(eta*z - 2*eta*l*vol*sqrt(tau))))
    return ret
