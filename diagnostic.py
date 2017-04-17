from scripts.classes import Option, Future
from scripts.portfolio import Portfolio
brokerage = 1
from math import log, sqrt, exp
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
    'C':   [0.393678571428571, 127.007166832986, 0.25, 10, 50],
    'BO':  [22.046, 27.215821, 0.01, 0.5, 600],
    'LC':  [22.046, 18.143881, 0.025, 1, 400],
    'LRC': [1, 10, 1, 50, 10],
    'KW':  [0.3674333, 136.07911, 0.25, 10, 50],
    'SM':  [1.1023113, 90.718447, 0.1, 5, 100],
    'COM': [1.0604, 50, 0.25, 2.5, 53.02],
    'OBM': [1.0604, 50, 0.25, 1, 53.02],
    'MW':  [0.3674333, 136.07911, 0.25, 10, 50]
}


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

    # initializing constants
    if vol == 0:
        print('vol is zero, setting small value.')
        vol = 0.0001
    assert tau > 0
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


def _compute_value(char, tau, vol, K, s, r, payoff, ki=None, ko=None, barrier=None, d=None, product=None):
    '''Wrapper function that computes value of option.
    Inputs: 1) ki      : Knock in value.
            2) ko      : Knock out value.
            3) barrier : type of barrier
            4) d       : direction of barrier.
            5) product : the underlying product
            #) Remaining inputs are identical to _bsm_euro.

    Outputs: Price of the option
    '''
    # expiry case
    if tau <= 0:
        return max(s-K, 0) if char == 'call' else max(K-s, 0)
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


def A_B(flag, phi, b, r, x1, x2, tau, vol, s, k):
    x = x1 if flag == 'A' else x2
    ret = phi*s*exp((b-r)*tau)*norm.cdf(phi*x) - phi*k * \
        exp(-r*tau)*norm.cdf(phi*x - phi*vol*sqrt(tau))
    return ret


def C_D(flag, phi, s, b, r, tau, h, mu, eta, y1, y2, k, vol):
    y = y1 if flag == 'C' else y2
    ret = (phi*s*exp((b-r)*tau)*((h/s)**(2*(mu + 1)))*norm.cdf(eta*y)) - \
        phi*k*exp(-r*tau) * ((h/s)**(2*mu))*norm.cdf(eta*y - eta*vol*sqrt(tau))
    return ret


def E_f(k, r, tau, eta, x, vol, h, s, mu, y):
    ret = k*exp(-r*tau) * (norm.cdf(eta*x - eta*vol*sqrt(tau)) -
                           ((h/s)**(2*mu)) * norm.cdf(eta*y - eta*vol*sqrt(tau)))
    return ret


def F_f(k, h, s, mu, l, eta, z, vol, tau):
    ret = k * (((h/s)**(mu + l)) * norm.cdf(eta*z) + (h/s)**(mu-l)
               * norm.cdf(eta*z - 2*eta*l * vol*sqrt(tau)))
    return ret


def _euro_barrier_amer_greeks(char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots, rebate=0):
    """Computes greeks of european options with american barriers. 

    Args:
        char (str)           : Call or Put.
        tau (double)         : time to expiry in years
        vol (double          : volatility.
        k (double)           : strike
        s (double)           : price of underlying
        r (double)           : interest rate
        payoff (str)         : american or european exercise. 'amer' or 'euro.'
        direction (str)      : direction of the barrier. 'up' or 'down'
        product (str)        : underlying product. eg: 'C'
        ki (double)          : knock-in value.
        ko (double)          : knock-out value
        lots (double)        : number of lots.
        rebate (int, optional): payback if option fails to knock in / knocks out.
    Returns:
        delta, gamma, theta, vega : greeks of this instrument.
    """
    # ticksize = multipliers[product][2]
    change_spot = 0.0001
    # change_spot = 0.01*ticksize
    change_vol = 0.01
    # change_vol = 0.0001
    change_tau = 1/365
    # change_tau = 1/(24/365)
    # computing delta
    # char, tau, vol, k, s, r, payoff, direction, ki, ko, rebate=0
    del1 = _barrier_amer(char, tau, vol, k, s+change_spot,
                         r, payoff, direction, ki, ko)
    del2 = _barrier_amer(char, tau, vol,
                         k, max(0, s-change_spot),
                         r, payoff, direction, ki, ko)
    delta = (del1 - del2)/(2*change_spot)

    # computing gamma
    del3 = _barrier_amer(char, tau, vol, k, s,
                         r, payoff, direction, ki, ko)
    gamma = (del1 - 2*del3 + del2) / ((change_spot**2))

    # g1 = _barrier_amer(char, tau, vol, k, s+(change_spot),
    #                    r, payoff, direction, ki, ko)
    # g2 = _barrier_amer(char, tau, vol, k, max(0, s - change_spot),
    #                    r, payoff, direction, ki, ko)
    # g3 = _barrier_amer(char, tau, vol, k, s,
    #                    r, payoff, direction, ki, ko)
    # g4 = _barrier_amer(char, tau, vol, k, s + 2*change_spot,
    #                    r, payoff, direction, ki, ko)
    # g5 = _barrier_amer(char, tau, vol, k, max(0, s - 2*change_spot),
    #                    r, payoff, direction, ki, ko)
    # gamma = (-g5 + 16*g2 - 30*g3 + 16*g1 - g4)/(12*(change_spot**2))

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
    ctau = 0.0001 if tau-change_tau <= 0 else tau-change_tau
    t2 = _barrier_amer(char, ctau, vol, k, s, r,
                       payoff, direction, ki, ko)
    theta = (t2 - t1)/change_tau if tau > 0 else 0
    # scaling greeks to retrieve dollar value.
    delta, gamma, theta, vega = greeks_scaled(
        delta, gamma, theta, vega, product, lots)
    return delta, gamma, theta, vega


def greeks_scaled(delta1, gamma1, theta1, vega1, product, lots):
    """Summary: Scaling method to bring greeks into reportable units.
    """

    lots = lots
    lm = multipliers[product][1]
    dm = multipliers[product][0]
    pnl_mult = multipliers[product][-1]
    delta = delta1 * lots
    gamma = (gamma1*lots*lm)/(dm)
    vega = (vega1*lots*pnl_mult)/100
    theta = (theta1*lots*pnl_mult)/365

    # return delta1, gamma1, theta1/365, vega1/100
    return delta, gamma, theta, vega


# char, tau, vol, k, s, r, payoff, direction, ki, ko, rebate=0
char, tau, vol, bvol, k, s, r, payoff, direction, product, ki, ko, lots = 'call', 68 / \
    365, 0.222812797, 0.2327, 380, 377.375, 0, 'euro', 'up', 'C', 400, None, 1000


x = _euro_barrier_amer_greeks(
    char, tau, vol, k, s, r, payoff, direction, product, ki, ko, lots)
