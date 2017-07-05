"""
File Name      : classes.py
Author         : Ananth Ravi Kumar
Date created   : 7/3/2017
Last Modified  : 4/5/2017
Python version : 3.5
Description    : Script contains implementation of the Option and Futures classes, as well as helper methods that set/store/manipulate instance variables. This class is used in simulation.py.

"""

# lots = 10
import numpy as np


class Option:

    """
    Definitions for the option class.
    Instance variables:
        1)  ki         =   knock-in value. Defaults to None if not used.
        2)  ko         =   knock-out value Defaults to None if not used. 
        3)  K          =   strike price
        4)  price      =   price according to BSM 
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
        22) ordering   =   indicating if this option is c1, c2, ... c8. For vol/price purposes.
        23) settlement =   string indicating if the option is cash- or future-settled upon exercise

        Notes: 
        1_ ki, ko, bullet, direc and barrier default to None and must be expressly overridden if an exotic option is desired.
        2_ knockedin and knockedout should be set ONLY for options that are knockin and knockout options respectively. the program assumes this assignment is done correctly.

    Instance Methods:
    1) get_month       : returns the month of the underlying contract.
    2) init_greeks     : initializes the greeks of this option. only called on init.
    3) update_greeks   : updates and reassigns greeks upon changes in underlying price/vol.
    4) greeks          : getter method that returns a tuple of the greeks
    5) compute_vol     : computes the implied volatility of this option.
    6) compute_price   : computes value of the object based on the appropriate valuation method.
    7) get_price       : getter method for the value of the object.
    8) update_tau      : updates time to expiry.
    9) exercise        : exercises the option. as of now, returns a boolean.
    10) moneyness      : returns 1, 0 , -1 depending if option is itm, atm, otm.
    11) get_underlying : returns the underlying future object.
    12) get_desc       : returns 'option'
    13) get_product    : returns the NAME of the underlying future.
    14) check_active   : for barriers, checks to see if this barrier is active or not.
    15) check_expired  : expires this option. defaults to false upon initialization.
    16) zero_option    : zeros the option's greeks.
    17) shorted        : indicates if this is security is being longed or shorted.
    18) set_ordering   : sets the ordering of this option.
    19) get_ordering   : return self.ordering
    20) get_month      : returns the month associated with this option.
    21) get_properties : returns a dictionary containing all the non-default properties of the option. Used for daily to bullet conversion. 
    """

    def __init__(self, strike, tau, char, vol, underlying, payoff, shorted, month, direc=None, barrier=None, lots=1000, bullet=True, ki=None, ko=None, rebate=0, ordering=1e5, settlement='futures', bvol=None):
        self.month = month
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
        self.bvol = bvol
        self.vol = vol
        self.r = 0
        self.shorted = shorted
        self.price = self.compute_price()
        self.ordering = ordering
        self.init_greeks()
        self.active = self.check_active()
        self.expired = False  # defaults to false.
        self.rebate = rebate
        self.product = self.get_product()
        self.settlement = settlement

    def __str__(self):
        string = '<<'
        string += self.product + ' ' + self.month + \
            '.' + self.underlying.get_month() + ' '

        string += str(self.K) + ' '
        if self.barrier is None:
            string += 'Vanilla '
        elif self.barrier == 'euro':
            string += 'E'
        string += 'C' if self.char == 'call' else 'P'
        if self.ki:
            if self.direc == 'up':
                string += 'UI '
            if self.direc == 'down':
                string += 'DI '
            string += str(self.ki)
        if self.ko:
            if self.direc == 'up':
                string += 'UO'
            if self.direc == 'down':
                string += 'DO'
            string += ' ' + str(self.ko)
        string += ' S ' if self.shorted else ' L '
        string += str(self.underlying.get_price())
        string += ' lots - ' + str(int(self.lots)) + ' |'
        string += ' ttm - ' + str(round(self.tau * 365)) + ' |'
        string += ' order - [c_' + str(self.ordering) + '] |'
        string += ' price - ' + str(self.price) + ' |'
        string += ' delta - ' + str(abs(self.delta)) + ' |'
        string += ' vol - ' + str(self.vol) + ' |'
        string += ' bvol - '
        string += str(self.bvol) if self.bvol is not None else 'None'
        string += '>>'
        return string

    def set_ordering(self, val):
        self.ordering = val

    def get_ordering(self):
        return self.ordering

    def decrement_ordering(self, i):
        self.ordering -= i
        # check expiration
        self.expired = self.check_expired()

    def get_op_month(self):
        return self.month

    def update_bvol(self, vol):
        self.bvol = vol

    def check_active(self):
        """Checks to see if this option object is active, i.e. if it has any value/contributes greeks. Cases are as follows:
        1) Knock-in barrier options are considered always active until expiry.
        2) Knock-out options with an american barrier are considered NOT active when they hit the barrier.
        3) Knock-out options with a European barrier are considered always active until expiry.
        4) Vanilla options are always active until expiry.
         """
        s = self.underlying.get_price()
        expired = self.check_expired()
        # expired case
        if expired:
            return False
        # base cases: if already knocked in or knocked out, return
        # appropriately.
        if self.knockedin:
            return True
        if self.knockedout:
            # first case: check for expiry.
            if self.tau == 0:
                return False
            # second: american barrier. ko = deactivate.
            elif self.barrier == 'amer':
                return False
            # final case: Euro barrier. active till exp.
            else:
                return True

        # barrier cases
        if self.ki:
            # all knockin options contribute greeks/have value until expiry.
            active = True if not expired else False
            if self.direc == 'up':
                self.knockedin = True if (s >= self.ki) else False
            if self.direc == 'down':
                self.knockedin = True if (s <= self.ki) else False
            if self.knockedin:
                self.ki, self.ko, self.barrier, self.direc = None, None, None, None
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
                active = True if not expired else False
                # european up and out
                if self.direc == 'up':
                    # print('Euro Up Out Hit')
                    self.knockedout = True if (s >= self.ko) else False
                # european down and out
                if self.direc == 'down':
                    self.knockedout = True if (s <= self.ko) else False
        else:
            # vanilla case. true till expiry
            active = True if not expired else False
        return active

    def get_underlying(self):
        return self.underlying

    def get_month(self):
        return self.underlying.get_month()

    def get_desc(self):
        return self.desc

    def init_greeks(self):
        from .calc import _compute_greeks
        # initializes relevant greeks. only used once, when initializing Option
        # object.
        product = self.get_product()
        # print(product)
        s = self.underlying.get_price()
        # print(s)
        try:
            assert self.tau > 0
            delta, gamma, theta, vega = _compute_greeks(
                self.char, self.K,  self.tau, self.vol, s, self.r, product, self.payoff, self.lots, ki=self.ki, ko=self.ko, barrier=self.barrier, direction=self.direc, order=self.ordering, bvol=self.bvol)
        except TypeError:
            print('char: ', self.char)
            print('strike: ', self.K)
            print('tau: ', self.tau)
            print('vol: ', self.vol)
            print('s: ', s)
            print('r: ', self.r)
            print('product: ', product)
            print('payoff: ', self.payoff)
            print('ki: ', self.ki)
            print('ko: ', self.ko)
            print('barrier: ', self.barrier)
            print('direction: ', self.direc)

        if self.shorted:
            # print('shorted!')
            delta, gamma, theta, vega = -delta, -gamma, -theta, -vega

        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        # return delta, gamma, theta, vega

    def update_greeks(self, vol=None, bvol=None):
        from .calc import _compute_greeks
        # method that updates greeks given new values of s, vol and tau, and subsequently updates value.
        # used in passage of time step.
        sigma, b_sigma = None, None
        active = self.check_active()
        self.active = active
        if active:
            sigma = vol
            b_sigma = bvol
            if vol is None:
                sigma = self.vol
            if bvol is None:
                b_sigma = self.bvol
            product = self.get_product()
            s = self.underlying.get_price()
            delta, gamma, theta, vega = _compute_greeks(
                self.char, self.K, self.tau, sigma, s, self.r, product, self.payoff, self.lots, ki=self.ki, ko=self.ko, barrier=self.barrier, direction=self.direc, order=self.ordering, bvol=b_sigma)
            # account for shorting
            if self.shorted:
                delta, gamma, theta, vega = -delta, -gamma, -theta, -vega

            self.delta, self.gamma, self.theta, self.vega = delta, gamma, theta, vega

            self.vol = sigma

            self.price = self.compute_price()
        else:
            self.zero_option()

    def greeks(self):
        # getter method for greeks. preserves abstraction barrier. updates just
        # in case price of underlying has changed.
        self.update_greeks()
        return self.delta, self.gamma, self.theta, self.vega

    def compute_vol(self, underlying, price, strike, tau, r):
        # computes implied vol from market price data. only holds for vanilla
        # options.
        from .calc import _compute_iv
        product = self.get_product()
        if self.barrier is None:
            return _compute_iv(underlying, price, strike, tau, r, product)

    def compute_price(self):
        from .calc import _compute_value
        # computes the value of this structure from relevant information.
        s = self.underlying.get_price()
        product = self.underlying.get_product()
        val = _compute_value(self.char, self.tau, self.vol, self.K, s, self.r,
                             self.payoff, ki=self.ki, ko=self.ko, barrier=self.barrier,
                             d=self.direc, product=product)
        return val

    def get_price(self):
        # check for expiry case
        if self.tau <= 0 or np.isclose(self.tau, 0):
            mult = -1 if self.shorted else 1
            s = self.underlying.get_price()
            k = self.K
            if self.char == 'call':
                val = max(s - k, 0)
            elif self.char == 'put':
                val = max(k - s, 0)
            return mult * val
        # not expired; check if active.
        else:
            active = self.check_active()
            self.active = active
            if self.active:
                price = self.compute_price()
                self.price = price
                return self.price
            else:
                return 0

    def update_tau(self, diff):
        self.tau -= diff

    def get_product(self):
        return self.underlying.get_product()

    def exercise(self):
        worth = self.moneyness()
        if (worth is not None) and (worth > 0):
            return True
        else:
            return False

    def moneyness(self):
        """Checks to see the 'moneyness' of the option.

        Returns:
            int: returns 1, 0, -1 for ITM, ATM and OTM options respectively, pr None if check_active() returns False 
        """
        active = self.check_active()
        # degenerate case: knocked out.
        if self.knockedout:
            return None
        self.active = active
        print('active: ', active)
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
                print('putop')
                print(self.K, s)
                # ITM
                if self.K > s:
                    return 1
                # OTM
                else:
                    return -1
        else:
            # print('hit none case')
            return -np.inf

    def zero_option(self):
        self.delta, self.gamma, self.theta, self.vega = 0, 0, 0, 0

    def check_expired(self):
        ret = True if (self.tau <= 0 or self.ordering <= 0) else False
        self.expired = ret
        return ret

    def update_lots(self, lots):
        self.lots = lots
        self.update_greeks()

    def get_properties(self):
        return {'month': self.month, 'barrier': self.barrier, 'payoff': self.payoff,
                'underlying': self.underlying, 'lots': self.lots, 'ki': self.ki,
                'ko': self.ko, 'direc': self.direc, 'strike': self.K, 'char': self.char,
                'vol': self.vol,  'shorted': self.shorted, 'ordering': self.ordering,
                'rebate': self.rebate, 'bvol': self.bvol, 'settlement': self.settlement}


class Future:

    '''
    Class representing a Future object. Instance variables are:
    1) month     :  the contract month.
    2) price     :  the quoted price of the future.
    3) desc      :  string description of the object
    4) lots      :  number of lots represented by each future contract.
    5) product   :  the commodity of this future.
    6) shorted   :  bool indicating whether this future is being shorted or \
    longed

    Instance Methods:
    1) get_desc       : returns 'future'
    2) get_price      : returns price of the future.
    3) update_price   : updates the price based on inputted data.
    5) get_month      : returns contract month.
    6) get_lots       : returns lot size
    7) get_product    : returns the name of this contract (i.e. the commodity)
    '''

    def __init__(self, month, price, product, shorted=None, lots=1000, ordering=None):
        self.product = product
        self.ordering = ordering
        self.lots = lots
        self.desc = 'future'
        self.month = month
        self.shorted = shorted
        if price >= 0:
            self.price = price
        else:
            raise ValueError("Price cannot be negative")
        self.expired = self.check_expired()
        mult = -1 if shorted else 1
        self.delta = 1 * lots * mult

    def __str__(self):
        string = self.product + ' ' + self.month + ' '
        string += str(self.price)
        string += ' S' if self.shorted else ' L'
        string += ' ' + str(int(self.lots))
        string += ' [c_' + str(self.ordering) + ']'
        return string

    def get_ordering(self):
        return self.ordering

    def set_ordering(self, i):
        self.ordering = i

    def decrement_ordering(self, i):
        self.ordering -= i

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

    def check_expired(self):
        ret = True if self.ordering == 0 else False
        self.expired = ret
        return ret

    def update_lots(self, lots):
        self.lots = lots
        mult = -1 if self.shorted else 1
        self.delta = 1 * lots * mult
