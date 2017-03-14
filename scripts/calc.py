"""
Script that contains implementation of the following calculation-related methods:

1) Pricing for various instruments:
    > vanilla options
    > barrier options

3) Calculating IV
4) Calculating PnL
 """


# product : [futures_multiplier - (dollar_mult, lot_mult),
# options_multiplier - (dollar_mult, lot_mult), futures_tick,
# options_tick, brokerage]

from math import log, sqrt, exp, pi
from scipy.stats import norm


#####################################################################
##################### Option pricing formulas #######################
#####################################################################


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


def _amer_option(option, tau, vol, K, s, r):
    """Vanilla american option pricing. 

    Inputs: 1) option     : call or put.
            2) tau        : time to expiry in years.
            3) vol        : volatility (sigma)
            4) K          : strike price
            5) underlying : price of underlying
            6) interest   : interest rate

    Output: 1) Price      : price of option according to BSM
    """
    pass


def _compute_value(char, tau, vol, K, s, r, payoff, ki=None, ko=None, barrier=None, dir=None):
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
            return _amer_option(char, tau, vol, K, s, r)
        elif payoff == 'euro':
            return _bsm_euro(char, tau, vol, K, s, r)
    # barrier option case
    else:
        if barrier == 'amer':
            return _barrier_amer(char, tau, vol, K, s, r, payoff, d)
        elif barrier == 'euro':
            return _barrier_euro(char, tau, vol, K, s, r, payoff, d)


def _barrier_euro(char, tau, vol, k, s, r, payoff, direction):
    """Inputs:
    1) Char      : call or put.
    2) tau       : time to expiry.
    3) vol       : volatility
    4) k         : strike price
    5) s         : price of underlying
    6) r         : interest rate
    7) payoff    : european or american option
    8) direction : direction of the barrier.

    Outputs:
    1) Price
    2) Delta
    3) Gamma
    4) Theta
    5) Vega

    """
    pass


def _barrier_amer(char, tau, vol, k, s, r, payoff):
    """Inputs:
    1) Char   : call or put.
    2) tau    : time to expiry.
    3) vol    : volatility
    4) k      : strike price
    5) s      : price of underlying
    6) r      : interest rate
    7) payoff : european or american option

    Outputs:
    1) Price  
    2) Delta
    3) Gamma
    4) Theta
    5) Vega

    """

    pass

#####################################################################
##################### Greek-related formulas ########################
#####################################################################

    # TODO: grab appropriate product-specific multipliers from dictionary and
    # feed into computation.


def _compute_greeks(char, K, tau, vol, s, r, product, payoff, barrier=None):
    """ Computes the greeks of various option profiles.
    Inputs:  1) char   : call or put
             2) K      : strike
             3) tau    : time to expiry
             4) vol    : volatility (sigma)
             5) s      : price of underlying
             6) r      : interest
             7) product: underlying commodity.
             8) payoff : american or european option.
             9) barrier: american or european barrier.

    Outputs: 1) delta  : dC/dS
             2) gamma  : d^2C/dS^2
             3) theta  : dC/dt
             4) vega   : dC/dvol
    """

    # european options
    if payoff == 'euro':
        # vanilla case
        if barrier is None:
            return _euro_vanilla_g(
                char, K, tau, vol, s, r, product)
        elif barrier = 'amer':
            # greeks for european options with american barrier.
            return _euro_barrier_amer_g()
        elif barrier = 'euro':
            # greeks for european options with european barrier.
            return _euro_barrier_euro_g()

    # american options
    elif payoff == 'amer':
        # vanilla case
        if barrier is None:
            return _amer_vanilla_g(
                char, K, tau, vol, s, r, product)
        # american options with american barrier
        elif barrier = 'amer':
            return _amer_barrier_amer_g()
        # american option with european barrier.
        elif barrier = 'euro':
            return _amer_barrier_euro_g()


def _euro_vanilla(char, K, tau, vol, s, r, product):
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


def _amer_vanilla_g(char, K, tau, vol, s, r, product):
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
    return _CRRbinomial('greeks', 'amer', char, s, K, tau, r, r, vol, product)


# TODO: _euro_barrier_amer_g()


def _euro_barrier_amer_g():
    """Computes greeks of european options with american barriers. """
    pass

# TODO: _euro_barrier_euro_g()


def _euro_barrier_euro_g():
    """Computes greeks of european options with european barriers. """
    pass


# TODO: _amer_barrier_euro_g


def _amer_barrier_euro_g():
    """Computes greeks of american options with european barriers. """
    pass

# TODO: _amer_barrier_amer_g


def _amer_barrier_amer_g():
    """Computes greeks of american options with american barriers. """
    pass


def _compute_vol(optiontype, s, k, c, tau, r, flag):
    """Computes implied volatility of plain vanilla puts and calls.
    Inputs:
    1) optiontype : call or put.
    2) s          : underlying
    3) k          : strike
    4) c          : option price 
    5) tau        : time to expiry
    6) flag       : american or european option
    7) barrier    : american or european barrier
    8) direction  : direction of barrier
    9) ki, ko     : knockin, knockout"""

    # european option
    if flag == 'euro':
        return newton_raphson(s, k c, tau, r, optiontype)
    # american option
    elif flag == 'amer':
        return american_iv(s, k c, tau, r, optiontype)

#############################################################
##### Supplemental Methods for Numerical Approximations #####
#############################################################


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
    precision = 1e-5
    guess = 0.5
    for i in range(num_iter):
        d1 = (log(s/K) + (r + 0.5 * guess ** 2)*tau) / \
            (guess * sqrt(tau))
        option_price = _bsm_euro(option, tau, guess, k, s, r)
        vega = s*(1/sqrt(2*pi)) * exp(-(d1**2) / 2) * sqrt(tau)
        price = option_price
        diff = c - option_price
        if abs(diff) < precision:
            return sigma
        sigma = sigma + diff/vega
    return sigma


def american_iv(s, k c, tau, r, optiontype):
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


def _CRRbinomial(output_flag, payoff, option_type, s, k, tau, r, b, vol, product, n=100):
    """Implementation of a Cox-Ross-Rubinstein Binomial tree. Translalted from The Complete Guide to Options Pricing Formulas, Chapter 7: Trees and Finite Difference methods by Espen Haug.

    Inputs:
        output_flag   : 'greeks' or 'price'
        payoff        : 'amer' or 'euro'
        option_type   : 'call' or 'put'
        s             : price of underlying
        k             : strike price
        tau           : time to expiry
        r             : interest rate
        b             : cost of carry
        vol           : volatility
        product       : the underlying product.
        n             : number of timesteps. defaults to 100.

    Returns:
        Price  : the price of this option
        Greeks : returns delta, gamma, theta, vega
    """

    # TODO: product specific multipliers.

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

    if output_flag = 'price':
        price = returnvalue[0]
        return price
    else:
        greeks = returnvalue[1:]
        vega = _num_vega(
            output_flag, payoff, option_type, s, k, tau, r, b, vol)
        greeks.append(vega)
        return greeks


def _num_vega(payoff, option_type, s, k, tau, r, b, vol):
    """Computes vega from CRR binomial tree if option passed in is American, and uses closed-form analytic formula for vega if option is European.

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
            'price', payoff, option_type, s, k, tau, r, b, vol+delta_v)
        lower = _CRRbinomial(
            'price', payoff, option_type, s, k, tau, r, b, vol-delta_v)
        vega = (upper - lower)/(2*delta_v)
    # european case. use analytic formulation for vega.
    elif payoff == 'euro':
        vega = s*(1/sqrt(2*pi)) * exp(-(d1**2) / 2) * sqrt(tau)
    return vega
