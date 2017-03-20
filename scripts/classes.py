"""
File Name      : classes.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 15/3/2017
Python version : 3.5
Description    : Script contains implementation of the Option and Futures classes, as well as helper methods that set/store/manipulate instance variables. This class is used in simulation.py.

"""


# File containing all the classes required by the simulation.
from .calc import _compute_value, _compute_greeks

lots = 10

# multipliers = prep.read_multipliers()

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


class Option:

    """
    Definitions for the option class.
    Instance variables:
        1)  ki         =   knock-in value.
        2)  ko         =   knock-out value
        3)  K          =   strike price
        4)  price      =   price quoted for the option.
        5)  tau        =   time to expiry in years (as per black-scholes)
        6)  char       =   call or put option.
        7)  vol        =   current implied volatility
        9)  r          =   risk free interest rate. assumed to be 0, can be changed.
        10) desc       =   string description of the object
        11) month      =   month of expiry. used to keep greeks for different months separate.
        12) barrier    =   American or European barrier.
        13) lots       =   the quantity of underlying being shorted or bought upon expiry. 
        14) bullet     =   True - Bullet. False - Daily.
        15) underlying =   the underlying futures object
        16) payoff     =   American or European option
        17) direc      =   Indicates the direction of the barrier (up and out etc.).
        18) knockedin  =   boolean indicating if this barrier knockin option is active.
        19) knockedout =   boolean indicating if this barrier knockout option is active.
        20) expired    =   boolean indicating if the option has expired.
        21) rebate     =   value paid to holder of option if option is knocked out.

        Notes: 
        1_ ki, ko, bullet, direc and barrier default to None and must be expressly overridden if an exotic option is desired.
        2_ knockedin and knockedout should be set ONLY for options that are knockin and knockout options respectively. the program assumes this assignment is done correctly.

    Instance Methods:
    1) get_month       : returns the month of the underlying contract.
    2) init_greeks     : initializes the greeks of this option. only called on init.
    3) update_greeks   : updates and reassigns greeks upon changes in underlying price/vol.
    4) greeks          : getter method that returns a tuple of the greeks
    5) compute_vol     :  [not implemented]
    6) compute_value   : computes value of the object based on the appropriate valuation method.
    7) get_value       : getter method for the value of the object.
    8) update_tau      : updates time to expiry.
    9) exercise        : exercises the option. as of now, returns a boolean.
    10) moneyness      : returns 1, 0 , -1 depending if option is itm, atm, otm.
    11) get_underlying : returns the underlying future object.
    12) get_desc       : returns 'option'
    13) get_product    : returns the NAME of the underlying future.
    14) check_active   : for barriers, checks to see if this barrier is active or not.
    15) expire         : expires this option. defaults to false upon initialization.
    """

    def __init__(self, strike, tau, char, vol, underlying, payoff, direc=None, barrier=None, lots=lots, bullet=None, ki=None, ko=None, rebate=0):
        self.barrier = barrier
        self.payoff = payoff
        self.underlying = underlying
        self.bullet = bullet  # daily = list of bullets.
        self.lots = lots
        self.desc = 'option'
        self.ki = ki
        self.ko = ko
        # defaults to None. Is set upon first check_active call.
        self.knockedout = None
        # defaults to None. Is set upon first check_active call.
        self.knockedin = None
        self.direc = direc
        self.K = strike
        self.tau = tau
        self.char = char
        self.vol = vol
        self.r = 0
        self.price = self.compute_price()
        self.delta, self.gamma, self.theta, self.vega = self.init_greeks()
        self.active = self.check_active()
        self.expired = False  # defaults to false.
        self.rebate = rebate
        self.product = self.get_product()

    def check_active(self):
        """Checks to see if this option object is active, i.e. if it has any value. Cases are as follows:
        1) Knock-in barrier options are considered always active until expiry.
        2) Knock-out options with an american barrier are considered NOT active when they hit the barrier.
        3) Knock-out options with a European barrier and considered always active until expiry.
        4) Vanilla options are always active until expiry.
         """
        s = self.underlying.get_price()
        cond = (self.tau > 0)
        if not cond:
            return False
        # base cases: if already knocked in or knocked out, return
        # appropriately.
        if self.knockedin:
            return True
        if self.knockedout:
            return False
        # barrier cases
        if self.ki:
            # all knockin options contribute greeks/have value until expiry.
            active = True if cond else False
            if self.direc == 'up':
                self.knockedin = True if (s >= self.ki) else False
            if self.direc == 'down':
                self.knockedin = True if (s <= self.ki) else False
        if self.ko:
            if self.barrier == 'amer':
                # american up and out
                if self.direc == 'up':
                    active = False if s >= self.ko else True
                    self.knockedout = not active
                # american down and out
                if self.direc == 'down':
                    active = False if s <= self.ko else True
                    self.knockedout = not active
            # european knockout are active until expiry.
            elif self.barrier == 'euro':
                active = True if cond else False
                # european up and out
                if self.direc == 'up':
                    self.knockedout = True if (s >= self.ko) else False
                # european down and out
                if self.direc == 'down':
                    self.knockedout = True if (s <= self.ko) else False
        else:
            # vanilla case. true till expiry
            active = True if cond else False
        return active

    def get_underlying(self):
        return self.underlying

    def get_month(self):
        return self.underlying.get_month()

    def get_desc(self):
        return self.desc

    def init_greeks(self):
        # initializes relevant greeks. only used once, when initializing Option
        # object.
        # char, K, tau, vol, s, r, product, payoff, lots, ki=None, ko=None,
        # barrier=None, direction=None
        product = self.get_product()
        s = self.underlying.get_price()
        return _compute_greeks(self.char, self.K,  self.tau, self.vol, s, self.r, product, self.payoff, self.lots, ki=self.ki, ko=self.ko, barrier=self.barrier, direction=self.direc)

    def update_greeks(self, vol=None):
        # method that updates greeks given new values of s, vol and tau, and subsequently updates value.
        # used in passage of time step.

        active = self.check_active()
        self.active = active
        if active:
            if vol is None:
                sigma = self.vol
            else:
                sigma = vol
            product = self.get_product()
            s = self.underlying.get_price()
            self.delta, self.gamma, self.theta, self.vega = _compute_greeks(
                self.char, self.K, self.tau, sigma, s, self.r, product, self.payoff, self.lots, ki=self.ki, ko=self.ko, barrier=self.barrier, direction=self.direc)
            self.vol = sigma
            self.price = self.compute_price()
        else:
            self.zero_option()

    def greeks(self):
        # getter method for greeks. preserves abstraction barrier. updates just
        # in case price of underlying has changed.
        self.update_greeks()
        return self.delta, self.gamma, self.theta, self.vega

    def compute_vol(underlying, price, strike, tau, r):
        # computes implied vol from market price data. only holds for vanilla
        # options.
        product = self.get_product()
        if self.barrier is None:
            return _compute_iv(underlying, price, strike, tau, r, product)

    def compute_price(self):
        # computes the value of this structure from relevant information.
        s = self.underlying.get_price()
        product = self.underlying.get_product()
        return _compute_value(self.char, self.tau, self.vol, self.K, s, self.r, self.payoff, ki=self.ki, ko=self.ko, barrier=self.barrier, d=self.direc, product=product)

    def get_price(self):
        # check for expiry case
        if self.tau == 0:
            s = self.underlying.get_price()
            k = self.K
            if self.char == 'call':
                return max(s - k, 0)
            elif self.char == 'put':
                return max(k - s, 0)
        # not expired; check if active.
        else:
            active = self.check_active()
            self.active = active
            if self.active:
                price = self.compute_price()
                self.price = price
                return self.price

    def update_tau(self, diff):
        self.tau -= diff

    def get_product(self):
        return self.underlying.get_product()

    def exercise(self):
        worth = self.moneyness()
        if worth > 0:
            return True
        else:
            return False

    def moneyness(self):
        active = self.check_active()
        self.active = active
        if active:
            s = self.underlying.get_price()
            # at the money
            if self.K == s:
                return 0
            if self.char == 'call':
                # ITM
                if self.K < s:
                    return 1
                # OTM
                else:
                    return -1
            elif self.char == 'put':
                # ITM
                if self.K > s:
                    return 1
                # OTM
                else:
                    return -1
        else:
            return None

    def zero_option(self):
        self.delta, self.gamma, self.theta, self.vega = 0, 0, 0, 0

    def check_expired(self):
        return True if self.tau == 0 else False


class Future:

    '''
    Class representing a Future object. Instance variables are:
    1) month  :  the contract month.
    2) price  :  the quoted price of the future.
    3) desc   :  string description of the object
    4) lots   :  number of lots represented by each future contract.
    5) product   :  the commodity of this future.

    Instance Methods:
    1) greeks         : dummy method.
    2) get_value      : returns price of the future.
    3) update_price   : updates the price based on inputted data.
    4) update_greeks  : dummy method.
    5) get_month      : returns contract month.
    6) get_lots       : returns lot size
    7) get_product    : returns the name of this contract (i.e. the commodity)
    '''

    def __init__(self, month, price, product, lots=lots):
        self.product = product
        self.lots = lots
        self.desc = 'future'
        self.month = month
        if price >= 0:
            self.price = price
        else:
            raise ValueError("Price cannot be negative")

    def get_price(self):
        return self.price

    def get_desc(self):
        return self.desc

    def update_price(self, price):
        # updates the price of the future object
        self.price = price

    def get_month(self):
        return self.month

    def get_lots(self):
        return self.lots

    def get_product(self):
        return self.product
